import torch

MINUTES_PER_DAY = 390.0


class StandardExecutionCostModel:
    """
    Standard execution cost model that combines multiple cost components:
      - Arrival cost
      - VWAP cost
      - Rate deviation penalty (expects base metric from env)
      - Holding risk cost
      - Unfilled cost (expects base metric from env)

    All data preparation (market fetches, cumulative VWAP, masks) should be done
    in the environment. This class only applies formulas and weights.
    """

    def __init__(
        self,
        arrival_cost_weight: float = 1.0,
        vwap_cost_weight: float = 1.0,
        rate_deviation_weight: float = 1.0,
        holding_risk_weight: float = 1.0,
        unfilled_cost_weight: float = 1.0,
        risk_lambda: float = 1.0,
    ) -> None:
        self.arrival_cost_weight = float(arrival_cost_weight)
        self.vwap_cost_weight = float(vwap_cost_weight)
        self.rate_deviation_weight = float(rate_deviation_weight)
        self.holding_risk_weight = float(holding_risk_weight)
        self.unfilled_cost_weight = float(unfilled_cost_weight)
        self.risk_lambda = float(risk_lambda)

    def get_total_cost(
        self,
        *,
        side: torch.Tensor,
        order_qty: torch.Tensor,
        arrival_price: torch.Tensor,
        fill_prices: torch.Tensor,
        trade_sizes: torch.Tensor,
        order_vwap: torch.Tensor,
        sigma_step: torch.Tensor,
        shares_remaining: torch.Tensor,
        adv: torch.Tensor,
        time_horizon: torch.Tensor,
        current_step: torch.Tensor,
        cum_market_volume: torch.Tensor,
        cum_market_dollars: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total cost and components.

        Returns (total_cost, components_dict)
        """
        # Arrival cost
        weight = trade_sizes / order_qty.clamp_min(1).float()
        price_perf = (fill_prices - arrival_price) / arrival_price
        arrival_cost = self.arrival_cost_weight * side * price_perf * weight

        market_ivwap = cum_market_dollars / cum_market_volume.clamp_min(1.0)

        # VWAP cost (slippage normalized by market IVWAP), only when trade executes
        vwap_slippage = (order_vwap - market_ivwap) / market_ivwap
        vwap_cost = self.vwap_cost_weight * side * vwap_slippage * weight
        vwap_cost = torch.where(trade_sizes > 0, vwap_cost, torch.zeros_like(vwap_cost))

        # Rate deviation penalty computed here
        # Actual completion ratio
        actual_completion_ratio = (
            (order_qty - shares_remaining).float() / order_qty.clamp_min(1).float()
        )
        # Target completion ratio using market volume to date and expected volume
        time_horizon_safe = time_horizon.clamp_min(1).float()
        volume_traded = cum_market_volume.float()
        expected_volume = adv.float() * (time_horizon_safe - current_step.float()) / MINUTES_PER_DAY
        denom = torch.clamp(volume_traded + expected_volume, min=1.0)
        target_completion_ratio = torch.clamp(volume_traded / denom, 0.0, 1.0)

        rate_deviation = (actual_completion_ratio - target_completion_ratio).abs()
        rate_penalty = self.rate_deviation_weight * sigma_step * rate_deviation

        # Holding risk cost
        holding_risk_cost = (
            self.holding_risk_weight
            * self.risk_lambda
            * sigma_step
            * (shares_remaining.float() / order_qty.clamp_min(1).float()).abs()
        )
        unfilled_base = torch.zeros_like(actual_completion_ratio)

        # Unfilled cost (base prepared in env)
        unfilled_cost = self.unfilled_cost_weight * unfilled_base

        total_cost = arrival_cost + vwap_cost + rate_penalty + holding_risk_cost + unfilled_cost

        components = {
            'arrival_cost': arrival_cost,
            'vwap_cost': vwap_cost,
            'rate_penalty': rate_penalty,
            'holding_risk_cost': holding_risk_cost,
            'unfilled_cost': unfilled_cost,
        }

        return total_cost, components


