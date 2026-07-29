<?php

# ─────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────
const ENV_ID          = "MountainCarContinuous-v0";
const SEED            = 42;
const TOTAL_STEPS     = 100000;
const START_STEPS     = 1000;
const BATCH_SIZE      = 256;
const BUFFER_SIZE     = 100000;
const LR_ACTOR        = 3e-4;
const LR_CRITIC       = 3e-4;
const LR_ALPHA        = 3e-4;
const GAMMA           = 0.99;
const TAU             = 0.005;
const HIDDEN_DIM      = 256;
const ALPHA_INIT      = 1.0;
const GSDE_LATENT_DIM = 64;
const GSDE_RESET_FREQ = 16;
const UPDATE_EVERY    = 1;
const EVAL_EVERY      = 1_000;
const EVAL_EPISODES   = 3;

