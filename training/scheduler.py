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
