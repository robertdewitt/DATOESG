import torch
import logging
from optimal_execution_env_vectorized import MINUTES_PER_DAY
import math

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
        cum_market_dollars: torch.Tensor,
        truncated_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total cost and components.

        Returns (total_cost, components_dict)
        """

        # order quantity should never be 0; at very least, use 1.
        oq = order_qty.clamp_min(1).float()

        # Arrival cost
        weight = trade_sizes.float() / oq
        # zero out any nans so they don't contribute to performance
        price_perf = torch.nan_to_num((fill_prices - arrival_price) / arrival_price, nan=0.0, posinf=0.0, neginf=0.0)
        arrival_cost = self.arrival_cost_weight * side.float() * price_perf * weight

        #logging.info(f"arrival_cost: {arrival_cost}, side: {side}, price_perf: {price_perf}, weight: {weight}, fill_prices: {fill_prices}, arrival_price: {arrival_price}, order_qty: {order_qty}, order_vwap: {order_vwap}, sigma_step: {sigma_step}, shares_remaining: {shares_remaining}, adv: {adv}, time_horizon: {time_horizon}, current_step: {current_step}, cum_market_volume: {cum_market_volume}, cum_market_dollars: {cum_market_dollars}")

        market_ivwap = torch.nan_to_num(cum_market_dollars / cum_market_volume.clamp_min(1.0),
                                nan=0.0, posinf=0.0, neginf=0.0)
        vwap_slippage = torch.nan_to_num((order_vwap - market_ivwap) / market_ivwap.clamp_min(1e-6),
                                 nan=0.0, posinf=0.0, neginf=0.0)
        vwap_cost = self.vwap_cost_weight * side.float() * vwap_slippage * weight
        vwap_cost = torch.where(trade_sizes > 0, vwap_cost, torch.zeros_like(vwap_cost))

        # Rate deviation penalty computed here
        # Actual completion ratio
        actual_completion_ratio = (
            (order_qty - shares_remaining).float() / oq
        )
        time_horizon_safe = time_horizon.clamp_min(1).float()
        expected_volume = adv.float() * (time_horizon_safe - current_step.float()) / float(MINUTES_PER_DAY)
        denom = torch.clamp(cum_market_volume.float() + expected_volume, min=1.0)
        target_completion_ratio = torch.clamp(cum_market_volume.float() / denom, 0.0, 1.0)
        actual_completion_ratio = (order_qty.float() - shares_remaining.float()) / oq
        rate_deviation = (actual_completion_ratio - target_completion_ratio).abs()
        rate_penalty = self.rate_deviation_weight * sigma_step * rate_deviation

        # Holding risk cost
        holding_risk_cost = (
            self.holding_risk_weight
            * self.risk_lambda
            * sigma_step
            * (shares_remaining.float() / oq).abs()
        )

        has_unfilled = truncated_mask & (shares_remaining > 0)
        unfilled_ratio = torch.zeros_like(order_vwap)
        unfilled_ratio = torch.where(has_unfilled,
                             shares_remaining.float() / oq,
                             unfilled_ratio)
        # prefer daily sigma; if only sigma_step is available:
        daily_sigma = sigma_step * math.sqrt(MINUTES_PER_DAY)
        unfilled_cost = getattr(self, "unfilled_penalty", 1.0) * (daily_sigma * unfilled_ratio)

        #logging.info(f"arrival_cost: {arrival_cost}, vwap_cost: {vwap_cost}, rate_penalty: {rate_penalty}, holding_risk_cost: {holding_risk_cost}, unfilled_cost: {unfilled_cost}")

        total_cost = arrival_cost + vwap_cost + rate_penalty + holding_risk_cost + unfilled_cost

        components = {
            'arrival_cost': arrival_cost,
            'vwap_cost': vwap_cost,
            'rate_penalty': rate_penalty,
            'holding_risk_cost': holding_risk_cost,
            'unfilled_cost': unfilled_cost,
        }

        return total_cost, components


