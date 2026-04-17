# Diverse Approaches to Optimal Execution Schedule Generation Codebase

This is the code for the following paper:

Diverse Approaches to Optimal Execution Schedule Generation
Robert de Witt, Mikko S. Pakkanen
We present the first application of MAP-Elites, a quality-diversity algorithm, to trade execution. Rather than searching for a single optimal policy, MAP-Elites generates a diverse portfolio of regime-specialist strategies indexed by liquidity and volatility conditions. Individual specialists achieve 8-10% performance improvements within their behavioural niches, while other cells show degradation, suggesting opportunities for ensemble approaches that combine improved specialists with the baseline PPO policy. Results indicate that quality-diversity methods offer promise for regime-adaptive execution, though substantial computational resources per behavioural cell may be required for robust specialist development across all market conditions.
To ensure experimental integrity, we develop a calibrated Gymnasium environment focused on order scheduling rather than tactical placement decisions. The simulator features a transient impact model with exponential decay and square-root volume scaling, fit to 400+ U.S. equities with R2>0.02 out-of-sample. Within this environment, two Proximal Policy Optimization architectures - both MLP and CNN feature extractors - demonstrate substantial improvements over industry baselines, with the CNN variant achieving 2.13 bps arrival slippage versus 5.23 bps for VWAP on 4,900 out-of-sample orders ($21B notional). These results validate both the simulation realism and provide strong single-policy baselines for quality-diversity methods.

https://arxiv.org/abs/2601.22113

To reproduce the results, the parquet version of the data is provided so one can run the models and play them back. 

Here are the steps to proceed:

1. Clone the repo from github
2. tar xvfz mana_data.tgz - this will unpack the parquet formed market data and analytics for the models to run against. (Note: at this phase we are not able to distribute this market data but this code can be reused with Yahoo data or if you write another handler other market data could be supported)
3. Inside the Gym directory, the primary model driver can be found: research_train_test_vec.ipynb
4. Run the cells sequentially and you should be able to reproduce the results as reported. All models are seeded so they should be deterministic. 
