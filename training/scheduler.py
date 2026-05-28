# training/scheduler.py
import math

class CycleScheduler:
    """
    Gère des rounds dont les budgets de phases (A = adv+mix, R = recon) croissent avec l'index de round.
    Les budgets s'expriment en 'unités epoch' (on incrémente d'1 par epoch consommée dans la phase).
    - base_adv, base_mix, base_rec : budgets initiaux
    - adv_boost, b_boost           : croissance linéaire par round
    - skip_amix                    : si True, mix=0
    """
    def __init__(self, *, base_adv: int, base_mix: int, base_rec: int,
                 adv_boost: int, b_boost: int, skip_amix: bool):
        self.base_adv  = int(base_adv)
        self.base_mix  = int(base_mix)
        self.base_rec  = int(base_rec)
        self.adv_boost = int(adv_boost)
        self.b_boost   = int(b_boost)
        self.skip_amix = bool(skip_amix)
        self.round = 0
        self._reset_round()

    def _budgets_for_round(self, r: int) -> tuple[int, int, int]:
        adv_k = self.base_adv + r * self.adv_boost
        mix_k = 0 if self.skip_amix else (self.base_mix + r * self.adv_boost)
        rec_k = self.base_rec + r * self.b_boost
        return max(0, adv_k), max(0, mix_k), max(0, rec_k)

    def _reset_round(self):
        self.adv_k, self.mix_k, self.rec_k = self._budgets_for_round(self.round)
        self.A_done = 0   # compte adv+mix (en epochs)
        self.R_done = 0

    def phase_now(self) -> str:
        # Renvoie la phase courante à l'intérieur du round
        if self.A_done < self.adv_k:
            return "A-adv"
        if self.A_done < (self.adv_k + self.mix_k):
            return "A-mix"
        return "B"

    def step_epoch(self):
        # Incrémente le compteur de la phase courante
        ph = self.phase_now()
        if ph.startswith("A"):
            self.A_done += 1
        else:
            self.R_done += 1

    def round_done(self) -> bool:
        return (self.A_done >= (self.adv_k + self.mix_k)) and (self.R_done >= self.rec_k)

    def next_round(self):
        self.round += 1
        self._reset_round()

    def budgets(self) -> dict:
        return {
            "adv": self.adv_k,
            "mix": self.mix_k,
            "rec": self.rec_k,
            "A_done": self.A_done,
            "R_done": self.R_done,
            "round": self.round,
        }

    # ----------------------------------------------------------------------
    # phase courante + lambdas (auto / hybrid)
    # ----------------------------------------------------------------------
    def current_phase_and_lambdas(
        self,
        λ_nce_AADV: float,
        λ_reg_AADV: float,
        λ_nce_AMIX: float,
        λ_reg_AMIX: float,
        λ_nce_B: float,
        λ_idt_B: float,
    ):
        """
        Renvoie (phase_courante, λN, λR) en fonction de la phase du cycle.
        """
        ph = self.phase_now()
        if ph == "A-adv":
            return ph, λ_nce_AADV, λ_reg_AADV
        if ph == "A-mix":
            return ph, λ_nce_AMIX, λ_reg_AMIX
        # phase B (reconstruction)
        return "B", λ_nce_B, λ_idt_B

    def current_lambdas(
        self,
        λ_nce_AADV: float,
        λ_reg_AADV: float,
        λ_nce_AMIX: float,
        λ_reg_AMIX: float,
        λ_nce_B: float,
        λ_idt_B: float,
    ):
        """
        Alias pratique : renvoie seulement (λN, λR) pour la phase courante.
        """
        _, λN, λR = self.current_phase_and_lambdas(
            λ_nce_AADV, λ_reg_AADV,
            λ_nce_AMIX, λ_reg_AMIX,
            λ_nce_B,   λ_idt_B,
        )
        return λN, λR



# =========================================================================================
#  ✅ SEM LR schedule: Warmup + Cosine Decay
#  -> À ajouter dans ce même fichier (ou dans un utils/scheduler.py)
# =========================================================================================

import math
import torch

def build_sem_warmup_cosine_scheduler(
    opt_sem: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    Crée un scheduler Warmup + Cosine decay pour l'optimizer SEM.

    - warmup_steps: nb de steps où lr monte de 0 → lr_base
    - total_steps: nb total de steps SEM (sur tout l'entraînement)
    - min_lr_ratio: lr_final = lr_base * min_lr_ratio

    Retourne: torch.optim.lr_scheduler.LambdaLR
    """
    total_steps = int(max(1, total_steps))
    warmup_steps = int(max(0, warmup_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int):
        step = int(step)
        # 1) warmup linéaire 0 -> 1
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # 2) cosine decay 1 -> min_lr_ratio
        if total_steps <= warmup_steps:
            return 1.0

        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)  # clamp [0,1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt_sem, lr_lambda=lr_lambda)


def sem_scheduler_get_lr(opt_sem: torch.optim.Optimizer) -> float:
    """Retourne le LR courant (1er groupe) de l'optimizer SEM."""
    try:
        return float(opt_sem.param_groups[0].get("lr", 0.0))
    except Exception:
        return 0.0


def ensure_sem_scheduler_in_state(state: dict, cfg: dict):
    """
    À appeler une seule fois après init(opt_SEM) (ou au premier sem step),
    pour créer et stocker le scheduler dans state.

    Clés cfg attendues (avec defaults robustes) :
      - sem_total_steps (int)       : total steps SEM sur tout le training
      - sem_warmup_steps (int)      : warmup steps SEM
      - sem_warmup_frac (float)     : si sem_warmup_steps absent, warmup = frac * total
      - sem_min_lr_ratio (float)    : ratio final
    """
    if state.get("opt_SEM", None) is None:
        return

    if state.get("sem_lr_sched", None) is not None:
        return  # déjà créé

    total_steps = int(cfg.get("sem_total_steps", 0))
    if total_steps <= 0:
        # fallback robuste (mieux vaut mettre sem_total_steps explicitement dans cfg)
        # On prend un ordre de grandeur pour ne pas casser:
        total_steps = int(cfg.get("total_steps", cfg.get("max_steps", 100000)))

    warmup_steps = cfg.get("sem_warmup_steps", None)
    if warmup_steps is None:
        warmup_frac = float(cfg.get("sem_warmup_frac", 0.05))
        warmup_steps = int(max(0, round(warmup_frac * total_steps)))
    warmup_steps = int(warmup_steps)

    min_lr_ratio = float(cfg.get("sem_min_lr_ratio", 0.1))

    sched = build_sem_warmup_cosine_scheduler(
        state["opt_SEM"],
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    state["sem_lr_sched"] = sched
    state["sem_lr_step"] = int(state.get("sem_lr_step", 0))
    state["sem_total_steps"] = total_steps
    state["sem_warmup_steps"] = warmup_steps
    state["sem_min_lr_ratio"] = min_lr_ratio


def step_sem_scheduler(state: dict):
    """
    Step le scheduler SEM (1 fois par update SEM).
    On garde un compteur séparé de steps SEM (sem_lr_step),
    car sem_every peut faire que SEM n'est pas mis à jour à chaque itération globale.
    """
    sched = state.get("sem_lr_sched", None)
    if sched is None:
        return
    # ⚠️ LambdaLR: step() incrémente d'un cran.
    sched.step()
    state["sem_lr_step"] = int(state.get("sem_lr_step", 0)) + 1

