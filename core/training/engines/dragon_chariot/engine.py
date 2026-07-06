"""DragonChariot engine: wires data pipeline, model, and training loop."""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from core.config.loader import instantiate_config, to_dict
from core.data.cache import ValCacheTap
from core.data.loader import (
    ValCacheSwapper,
    batch_to_device,
    build_dataloaders,
    build_pretrain_loader,
    finalize_val_cache,
)
from core.data.masking import FeatureMaskingScheduler
from core.models.dragon_chariot import DragonChariot
from core.training.callbacks.diagnostics import Diagnostics
from core.training.callbacks.run_writer import RunWriter
from core.training.checkpoint import CheckpointManager
from core.training.early_stopping import EarlyStopping
from core.training.ema import ModelEMA
from core.training.engines.dragon_chariot.loop import Trainer
from core.utils.device import TransparentDataParallel, TransparentDDP, select_device
from core.utils.distributed import get_rank_info
from core.utils.logging import SQLiteStore
from core.utils.seed import seed_everything

LOG = logging.getLogger(__name__)


# TODO: candidate for removal
def _print_memray_report(bin_path: str, top_n: int = 30) -> None:
    """Print top allocations from a memray capture file."""
    import os

    from memray import FileReader

    file_size_mb = os.path.getsize(bin_path) / 1024**2
    reader = FileReader(bin_path)
    records = list(reader.get_high_watermark_allocation_records(merge_threads=True))
    records.sort(key=lambda r: r.size, reverse=True)

    total = sum(r.size for r in records)
    LOG.info("=== Memray high-watermark report ===")
    LOG.info("File: %s (%.1f MB)", bin_path, file_size_mb)
    LOG.info("Peak tracked memory: %.2f GB (%d allocation sites)", total / 1024**3, len(records))
    LOG.info("Top %d:", top_n)
    for i, rec in enumerate(records[:top_n]):
        size_mb = rec.size / 1024**2
        frames = rec.stack_trace()
        if frames:
            fname, lineno, func = frames[0]
            # Shorten paths for readability
            short = fname.split("site-packages/")[-1] if "site-packages/" in fname else fname
            location = f"{short}:{lineno} in {func}"
        else:
            location = "<unknown>"
        LOG.info("  #%02d  %8.1f MB  %s", i + 1, size_mb, location)


_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Silence noisy third-party loggers
for _name in ("matplotlib", "tensorflow", "absl"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def setup_and_train(cfg: DictConfig) -> dict[str, float]:
    """Build everything from config and train.

    Parameters
    ----------
    cfg
        OmegaConf config with keys: train, model, data, diagnostics.
    """
    raw_cfg = cfg
    cfg = instantiate_config(to_dict(cfg))

    if "topo_lr" in cfg.train and cfg.train.topo_lr is not None:
        raise RuntimeError("train.topo_lr is deprecated.")
    if "compass" in cfg.train and cfg.train.compass is not None:
        raise RuntimeError("train.compass is deprecated.")

    # DDP: read rank/world_size from env (set by torchrun/TorchDistributor)
    local_rank, global_rank, world_size = get_rank_info()
    use_ddp = False
    if cfg.train.ddp:
        if world_size > 1:
            use_ddp = True
            LOG.info("Initializing distributed process group (rank %d/%d)", global_rank, world_size)
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        else:
            LOG.info("DDP enabled but world_size=1; running without DDP.")

    is_main = global_rank == 0

    # Output directory and logging (rank 0 only)
    run_dir = None
    file_handler = None
    if cfg.train.output_dir and is_main:
        run_dir = Path(cfg.train.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(raw_cfg, run_dir / "config.yaml")

        file_handler = logging.FileHandler(run_dir / "train.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.addHandler(file_handler)
        if root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)

    if not is_main:
        logging.getLogger().setLevel(logging.ERROR)

    seed_everything(cfg.train.seed)
    device = f"cuda:{local_rank}" if use_ddp else str(select_device(cfg.train.device))
    LOG.info("Device: %s (rank %d/%d, ddp=%s)", device, global_rank, world_size, use_ddp)

    data_cfg = dict(cfg.data)
    if "val_cache" in data_cfg:
        val_cache_cfg = data_cfg.pop("val_cache")
    else:
        val_cache_cfg = None
    if val_cache_cfg is not None and val_cache_cfg["enabled"] and cfg.train.mid_epoch_evals > 0:
        raise RuntimeError(
            "Validation cache tap requires train.mid_epoch_evals=0 "
            "(mid-epoch validation is unsupported when val cache is enabled)."
        )

    # Data — only shard across ranks when DDP is active
    train_loader, val_loader, dataset = build_dataloaders(
        **data_cfg,
        rank=global_rank if use_ddp else 0,
        world_size=world_size if use_ddp else 1,
    )
    if train_loader is None:
        raise RuntimeError(
            "Train split is empty. Check data.valid_ratio/train_ratio; "
            "setup_and_train requires a non-empty train loader."
        )
    fit_state = dataset.fit_state()
    val_loader_swapper = None

    schema = dataset.feature_schema
    LOG.info("Steps per epoch: %d", len(train_loader))

    # AMP
    amp_dtype = _AMP_DTYPES[cfg.train.amp_dtype] if cfg.train.amp_dtype else None
    LOG.info("AMP: %s", amp_dtype or "disabled")

    # Feature masking scheduler
    total_steps = len(train_loader) * cfg.train.max_epochs
    feature_masker = None
    if cfg.train.feature_masking:
        seq_domains = sorted(
            {s.domain for s in schema.query("scope = 'seq' and source != 'metadata'") if s.domain}
        )
        feature_masker = FeatureMaskingScheduler(
            schema=schema,
            seq_domains=seq_domains,
            total_steps=total_steps,
            epochs=cfg.train.max_epochs,
            **cfg.train.feature_masking,
        )

    # Model
    model = DragonChariot(
        schema=schema,
        feature_masker=feature_masker,
        **cfg.model,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_sparse = sum(p.numel() for p in model.get_sparse_params())
    n_dense = n_params - n_sparse
    LOG.info(
        "DragonChariot: %d params (%d sparse, %d dense), %d seq domains, format=%s",
        n_params,
        n_sparse,
        n_dense,
        len(model.domains),
        cfg.data.collator_type,
    )

    # DataParallel (legacy, mutually exclusive with DDP)
    use_dp = cfg.train.data_parallel and not use_ddp and torch.cuda.device_count() > 1
    if use_dp and cfg.train.enable_torch_compile:
        raise RuntimeError("DataParallel is not compatible with torch.compile")
    if use_dp:
        LOG.info("Wrapping model with DataParallel (%d devices)", torch.cuda.device_count())
        model = TransparentDataParallel(model)

    # torch.compile (before DDP wrap — DDP's DDPOptimizer handles graph breaks)
    if cfg.train.enable_torch_compile:
        warnings.filterwarnings("ignore", message="Online softmax is disabled on the fly")
        LOG.info("Enabling torch.compile on model")
        model.enable_compile()

    # DDP wrap (after compile) — only forward() routes through DDP for gradient sync
    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        LOG.info("Wrapping model with DDP (rank %d)", global_rank)
        # TODO(nsarang): keep broadcast_buffers=False when DDP is re-enabled;
        # diagnostic-only module buffers should not participate in DDP buffer sync.
        model = TransparentDDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    # Sequence domains (for eval per-domain metrics)
    seq_domains = sorted(
        {s.domain for s in schema.query("scope = 'seq' and source != 'metadata'") if s.domain}
    )

    # EMA
    ema = None
    if cfg.train.ema is not None:
        sparse_ptrs = set()
        if hasattr(model, "get_sparse_params"):
            sparse_ptrs = {p.data_ptr() for p in model.get_sparse_params()}
        steps_per_epoch = len(train_loader)
        ema = ModelEMA.from_config(cfg.train.ema, model, steps_per_epoch, skip_params=sparse_ptrs)

    # Pretrain loader (independent RNG, same dataset)
    pretrain_loader = None
    if cfg.train.pretrain_phase:
        pretrain_loader = build_pretrain_loader(train_loader)

    if val_cache_cfg is not None:
        LOG.info("Configuring validation cache tap with config: %s", val_cache_cfg)
        val_cache_tap = ValCacheTap(
            data_cfg=data_cfg,
            train_loader=train_loader,
            dataset=dataset,
            val_loader=val_loader,
            fit_state=fit_state,
            is_main=is_main,
            rank=global_rank if use_ddp else 0,
            world_size=world_size if use_ddp else 1,
            **val_cache_cfg,
        )
        if val_cache_tap.enabled and val_cache_tap.cache_hit:
            # Cache already built on a prior run — swap val loader immediately.
            val_loader = finalize_val_cache(
                cache_dir=val_cache_tap.cache_dir,
                data_cfg=data_cfg,
                fit_state=fit_state,
            )
        elif val_cache_tap.enabled:
            # Attach tap; swapper fires once after epoch 1, then becomes a no-op.
            val_cache_tap.attach(train_loader.dataset)
            val_loader_swapper = ValCacheSwapper(
                tap=val_cache_tap,
                data_cfg=data_cfg,
                fit_state=fit_state,
                use_ddp=use_ddp,
                local_rank=local_rank,
                global_rank=global_rank,
            )

        if val_cache_tap.enabled and not val_cache_tap.cache_hit and use_ddp:
            dist.barrier(device_ids=[local_rank])

    # Checkpoint manager
    checkpoint_mgr = None
    if run_dir and cfg.train.checkpoint:
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=Path(cfg.train.checkpoint.dir),
            config_dict=to_dict(raw_cfg),
            ckpt_name_template=cfg.train.checkpoint.name,
            schema_path=cfg.data.schema_path,
            amp_dtype=amp_dtype,
            fit_state=fit_state,
        )

    # Early stopping
    early_stopping = None
    if cfg.train.early_stopping:
        early_stopping = EarlyStopping(**cfg.train.early_stopping)

    # Observers (diagnostics + run writer) — rank 0 only
    observers = []

    if run_dir and is_main:
        store = SQLiteStore(run_dir)
        run_writer = RunWriter(store, config=cfg)
        if cfg.train.sqlite_store:
            observers.append(run_writer)

        if cfg.diagnostics is not None:
            log_dir = cfg.diagnostics.log_dir if cfg.diagnostics.tensorboard else None
            if cfg.diagnostics.get("n_logs_per_epoch") is None:
                n_emissions = max(1, round(len(train_loader) * cfg.diagnostics.step_log_rate))
            else:
                n_emissions = cfg.diagnostics.n_logs_per_epoch
            log_every_n_steps = max(1, len(train_loader) // n_emissions)
            active_codes = set(cfg.diagnostics.active_codes)
            diagnostics = Diagnostics(
                active_codes=active_codes,
                log_dir=log_dir,
                warmup_steps=cfg.diagnostics.warmup_steps,
                log_every_n_steps=log_every_n_steps,
                code_config=dict(cfg.diagnostics.code_config or {}),
                static_context={
                    "device": str(device),
                    "amp_dtype": amp_dtype,
                    "schema": schema,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "seq_domains": seq_domains,
                },
            )
            diagnostics.emit_preamble(
                seed=cfg.train.seed,
                config=raw_cfg,
                model=model,
                data_dir=cfg.data.dataset_path,
                schema_path=cfg.data.schema_path,
            )
            observers.append(diagnostics)

    # Loss function
    loss_fn = None
    if cfg.train.loss:
        loss_fn = instantiate_config(cfg.train.loss)

    dense_param_overrides = dict(cfg.train.dense_param_overrides or {})

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        loss_fn=loss_fn or torch.nn.BCEWithLogitsLoss(),
        dense_optimizer=cfg.train.dense_optimizer,
        sparse_optimizer=cfg.train.sparse_optimizer,
        dense_param_overrides=dense_param_overrides,
        lr_schedule=cfg.train.lr_schedule,
        epochs=cfg.train.max_epochs,
        grad_clip_norm=cfg.train.grad_clip,
        device=str(device),
        mid_epoch_evals=cfg.train.mid_epoch_evals,
        early_stopping=early_stopping,
        reinit=cfg.train.reinit,
        amp_dtype=amp_dtype,
        checkpoint_mgr=checkpoint_mgr,
        observers=observers,
        progress_bar=cfg.train.progress_bar,
        exclude_wd_on_bias_norm=cfg.train.exclude_wd_on_bias_norm,
        ema=ema,
        pretrain_phase=cfg.train.pretrain_phase,
        pretrain_loader=pretrain_loader,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        val_loader_swapper=val_loader_swapper,
    )

    memray_cfg = cfg.train.get("memray")
    if memray_cfg:
        import memray

        bin_path = memray_cfg.get("output", "/tmp/pp_memray.bin")
        follow_fork = memray_cfg.get("follow_fork", True)
        LOG.info("Memray profiling enabled: %s (follow_fork=%s)", bin_path, follow_fork)
        tracker = memray.Tracker(bin_path, follow_fork=follow_fork)
        tracker.__enter__()

    result = trainer.train()

    if memray_cfg:
        tracker.__exit__(None, None, None)
        _print_memray_report(bin_path, top_n=memray_cfg.get("top_n", 30))
        if memray_cfg.get("delete_after_report", False):
            os.unlink(bin_path)
            LOG.info("Deleted memray file: %s", bin_path)

    if use_ddp:
        dist.destroy_process_group()

    if file_handler:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    return result


def setup_and_infer(cfg: DictConfig) -> dict[str, list[float]]:
    """Load config + checkpoint, build model, run inference, write predictions.json.

    Parameters
    ----------
    cfg
        OmegaConf config with keys: train, model, data, infer.
    """
    cfg = instantiate_config(to_dict(cfg))

    seed_everything(cfg.train.seed)
    device = str(select_device(cfg.train.device))
    LOG.info("Device: %s", device)

    ckpt_path = CheckpointManager.resolve_path(Path(cfg.train.checkpoint.dir))
    LOG.info("Loading checkpoint: %s", ckpt_path)
    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Data — build in non-training mode; use ALL rows for inference (no train/val split)
    data_kwargs = dict(cfg.data)
    data_kwargs.pop("val_cache", None)
    data_kwargs["shuffle_train"] = False
    data_kwargs["is_training"] = False
    data_kwargs["valid_ratio"] = 1.0
    data_kwargs["fit_state"] = ckpt_state.get("fit_state")
    _, val_loader, dataset = build_dataloaders(**data_kwargs)
    schema = dataset.feature_schema

    eval_batch_size = cfg.train.eval_batch_size if hasattr(cfg.train, "eval_batch_size") else None
    if eval_batch_size:
        data_kwargs["batch_size"] = eval_batch_size
        _, val_loader, dataset = build_dataloaders(**data_kwargs)

    # Model (no feature masker at inference)
    model = DragonChariot(
        schema=schema,
        feature_masker=None,
        **cfg.model,
    ).to(device)

    # Apply model weights (strip _orig_mod prefix from sub-module compile)
    state_dict = ckpt_state.get("model_state_dict", ckpt_state)
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # Apply EMA weights if available
    if "ema_state_dict" in ckpt_state:
        sparse_ptrs = set()
        if hasattr(model, "get_sparse_params"):
            sparse_ptrs = {p.data_ptr() for p in model.get_sparse_params()}
        ema = ModelEMA(model, skip_params=sparse_ptrs)
        ema.load_state_dict(ckpt_state["ema_state_dict"])
        ema.apply()
        LOG.info("Loaded EMA weights for inference")
    model.eval()

    # AMP: default to fp32; only enable if use_trained_precision is set
    amp_dtype = None
    infer_cfg = cfg.infer
    if infer_cfg.use_trained_precision:
        saved_amp = ckpt_state.get("amp_dtype")
        if saved_amp is not None:
            amp_dtype = getattr(torch, saved_amp, None)
            if amp_dtype is not None:
                LOG.info("Inference AMP dtype from checkpoint: %s", amp_dtype)

    device_obj = torch.device(device)
    all_probs: list[float] = []
    all_user_ids: list[int] = []

    LOG.info("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            user_ids = batch["user_id"].cpu().numpy().tolist()
            batch = batch_to_device(batch, device_obj)

            if amp_dtype is not None:
                with torch.autocast(device_type=device_obj.type, dtype=amp_dtype):
                    logits, _ = model(batch)
                    logits = logits.reshape(-1)
            else:
                logits, _ = model(batch)
                logits = logits.reshape(-1)

            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                LOG.info("  Processed %d batches", batch_idx + 1)

    LOG.info("Inference complete: %d predictions", len(all_probs))

    # Write predictions as {user_id: probability} mapping
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"predictions": {str(uid): prob for uid, prob in zip(all_user_ids, all_probs)}}

    output_path = output_dir / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    LOG.info("Saved %d predictions to %s", len(all_probs), output_path)

    return predictions
