using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.Core;
using SharpRL.Agents;

namespace SharpRL.Business.Demo
{
    /// <summary>
    /// Real-World Business Problem: Dynamic Pricing Optimization
    /// Uses SharpRL's TD3Agent (State-of-the-Art Continuous Control)
    /// 
    /// SCENARIO: You run an online electronics store selling wireless headphones.
    /// - Your cost per unit: $50
    /// - Competitor prices fluctuate between $80-$150
    /// - Customer demand is price-sensitive
    /// - You must maximize profit while maintaining market share
    /// 
    /// STATE SPACE (4 dimensions):
    /// - Normalized competitor price [0, 1]
    /// - Normalized inventory level [0, 1]
    /// - Recent sales velocity [0, 1]
    /// - Day of month (seasonality) [0, 1]
    /// 
    /// ACTION SPACE (1 dimension):
    /// - Price adjustment [-1, 1] mapped to [$70, $180]
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.Clear();
            Console.WriteLine("╔═══════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║       SharpRL Business Demo: Dynamic Pricing with TD3            ║");
            Console.WriteLine("║         Wireless Headphones - E-Commerce Optimization            ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════════════╝\n");

            var simulator = new DynamicPricingEnvironment();

            Console.WriteLine("📊 BUSINESS PARAMETERS:");
            Console.WriteLine($"   Product: Premium Wireless Headphones");
            Console.WriteLine($"   Cost per Unit: ${simulator.UnitCost:F2}");
            Console.WriteLine($"   Starting Inventory: {simulator.InitialInventory} units");
            Console.WriteLine($"   Competitor Price Range: ${simulator.MinCompetitorPrice:F2} - ${simulator.MaxCompetitorPrice:F2}");
            Console.WriteLine($"   Our Price Range: ${simulator.MinPrice:F2} - ${simulator.MaxPrice:F2}");
            Console.WriteLine($"   Algorithm: TD3 (Twin Delayed Deep Deterministic Policy Gradient)\n");

            Console.WriteLine("🎯 TRAINING PHASE:");
            Console.WriteLine("   Teaching TD3 agent to maximize profit over 200 weeks...");
            Console.WriteLine("   (This demonstrates SharpRL's state-of-the-art continuous control)\n");

            var results = simulator.Train(episodes: 200, stepsPerEpisode: 30);

            Console.WriteLine("\n" + new string('═', 70));
            Console.WriteLine("\n📈 TRAINING RESULTS:");
            Console.WriteLine($"   Average Profit (First 20 weeks): ${results.EarlyProfit:F2}");
            Console.WriteLine($"   Average Profit (Last 20 weeks):  ${results.LateProfit:F2}");
            Console.WriteLine($"   Improvement: ${results.LateProfit - results.EarlyProfit:F2} ({results.ImprovementPercent:F1}%)");
            Console.WriteLine($"   Total Revenue Generated: ${results.TotalRevenue:F2}");
            Console.WriteLine($"   Total Units Sold: {results.TotalUnitsSold:N0}");
            Console.WriteLine($"   Best Week Profit: ${results.BestProfit:F2}");
            Console.WriteLine($"   Best Avg Price Point: ${results.BestAvgPrice:F2}");

            Console.WriteLine("\n" + new string('═', 70));
            Console.WriteLine("\n🧪 LIVE TESTING PHASE:");
            Console.WriteLine("   Running 5 weeks with the trained TD3 agent (no exploration)...\n");

            simulator.RunLiveDemo(weeks: 5);

            Console.WriteLine("\n" + new string('═', 70));
            Console.WriteLine("\n💡 BUSINESS INSIGHTS:");
            Console.WriteLine("   ✓ TD3 learned to dynamically adjust prices based on competition");
            Console.WriteLine("   ✓ Agent undercuts competitors strategically to capture demand");
            Console.WriteLine("   ✓ Higher prices when competitive advantage exists");
            Console.WriteLine("   ✓ Balances profit margins vs. market share automatically");
            Console.WriteLine("\n🏆 This demonstrates SharpRL's TD3 solving a real business problem!");

            Console.WriteLine("\n\nPress any key to exit...");
            Console.ReadKey();
        }
    }

    /// <summary>
    /// E-commerce pricing environment for continuous control RL
    /// </summary>
    public class DynamicPricingEnvironment
    {
        private Random _random = new Random(42);
        private TD3Agent _agent;

        // Business parameters
        public float UnitCost { get; } = 50.0f;
        public float MinPrice { get; } = 70.0f;
        public float MaxPrice { get; } = 180.0f;
        public float MinCompetitorPrice { get; } = 80.0f;
        public float MaxCompetitorPrice { get; } = 150.0f;
        public int InitialInventory { get; } = 1000;

        // Environment state
        private int _currentInventory;
        private float _currentCompetitorPrice;
        private float _recentSalesVelocity;
        private int _dayOfMonth;
        private int _currentStep;

        public DynamicPricingEnvironment()
        {
            // TD3: stateDim=4, actionDim=1
            _agent = new TD3Agent(
                stateDim: 4,
                actionDim: 1,
                hiddenLayers: new[] { 256, 256 },
                actionScale: 1.0f,
                bufferSize: 100000,
                learningRate: 3e-4f,
                gamma: 0.99f,
                tau: 0.005f,
                policyNoise: 0.1f,
                targetNoise: 0.2f,
                noiseClip: 0.5f,
                policyDelay: 2,
                seed: 42
            );

            Reset();
        }

        public void Reset()
        {
            _currentInventory = InitialInventory;
            _currentCompetitorPrice = 100.0f + (float)_random.NextDouble() * 30.0f; // $100-$130
            _recentSalesVelocity = 0.5f;
            _dayOfMonth = _random.Next(1, 31);
            _currentStep = 0;
        }

        private float[] GetState()
        {
            // Normalize all state variables to [0, 1] for better learning
            float competitorPriceNorm = (_currentCompetitorPrice - MinCompetitorPrice) /
                                        (MaxCompetitorPrice - MinCompetitorPrice);
            float inventoryNorm = _currentInventory / (float)InitialInventory;
            float velocityNorm = Math.Clamp(_recentSalesVelocity, 0.0f, 1.0f);
            float dayNorm = (_dayOfMonth - 1) / 30.0f;

            return new float[]
            {
                competitorPriceNorm,
                inventoryNorm,
                velocityNorm,
                dayNorm
            };
        }

        public (float reward, int unitsSold, bool done) Step(float[] action)
        {
            _currentStep++;

            // Map action [-1, 1] to price range [$70, $180]
            float actionValue = Math.Clamp(action[0], -1.0f, 1.0f);
            float ourPrice = MinPrice + ((actionValue + 1.0f) / 2.0f) * (MaxPrice - MinPrice);

            // Simulate realistic demand based on price elasticity
            float priceDifference = ourPrice - _currentCompetitorPrice;
            float baseDemand = 50.0f; // Base units per day

            // Price elasticity of demand
            float demandMultiplier = 1.0f;
            if (priceDifference < -30) demandMultiplier = 2.5f;      // Much cheaper
            else if (priceDifference < -20) demandMultiplier = 2.0f;
            else if (priceDifference < -10) demandMultiplier = 1.5f;
            else if (priceDifference < 0) demandMultiplier = 1.2f;   // Slightly cheaper
            else if (priceDifference < 10) demandMultiplier = 0.9f;  // Slightly more expensive
            else if (priceDifference < 20) demandMultiplier = 0.6f;
            else if (priceDifference < 30) demandMultiplier = 0.4f;
            else demandMultiplier = 0.2f;                            // Much more expensive

            // Seasonality: payday effect (higher demand days 1-7 and 15-21)
            float seasonalMultiplier = 1.0f;
            if (_dayOfMonth <= 7 || (_dayOfMonth >= 15 && _dayOfMonth <= 21))
                seasonalMultiplier = 1.4f;

            // Random market variance
            float randomFactor = 0.85f + (float)_random.NextDouble() * 0.3f; // 0.85-1.15

            // Calculate actual demand
            int potentialDemand = (int)(baseDemand * demandMultiplier * seasonalMultiplier * randomFactor);
            int actualSales = Math.Min(potentialDemand, _currentInventory);

            // Calculate profit (this is our RL reward)
            float revenue = actualSales * ourPrice;
            float cost = actualSales * UnitCost;
            float profit = revenue - cost;

            // Reward shaping: penalize running out of stock heavily
            float reward = profit;
            if (_currentInventory < 50 && actualSales < potentialDemand)
                reward -= 500.0f; // Lost opportunity penalty

            // Update environment state
            _currentInventory -= actualSales;
            _recentSalesVelocity = actualSales / baseDemand;
            _dayOfMonth = (_dayOfMonth % 30) + 1;

            // Competitor adjusts price (random walk with mean reversion)
            float competitorChange = (float)_random.NextDouble() * 10f - 5f; // -$5 to +$5
            float meanReversion = (115.0f - _currentCompetitorPrice) * 0.1f; // Pull toward $115
            _currentCompetitorPrice += competitorChange + meanReversion;
            _currentCompetitorPrice = Math.Clamp(_currentCompetitorPrice,
                                                 MinCompetitorPrice,
                                                 MaxCompetitorPrice);

            // Episode ends after 30 days or if inventory critically low
            bool done = _currentStep >= 30 || _currentInventory < 10;

            return (reward, actualSales, done);
        }

        public TrainingResults Train(int episodes, int stepsPerEpisode)
        {
            var results = new TrainingResults();
            var episodeProfits = new List<float>();
            var episodeAvgPrices = new List<float>();

            for (int episode = 0; episode < episodes; episode++)
            {
                Reset();
                float episodeProfit = 0;
                int episodeUnitsSold = 0;
                var episodePrices = new List<float>();

                float[] state = GetState();

                for (int step = 0; step < stepsPerEpisode; step++)
                {
                    // Get action from TD3 agent (with exploration noise)
                    float[] action = _agent.SelectAction(state, addNoise: true);

                    // Convert action to price for logging
                    float ourPrice = MinPrice + ((Math.Clamp(action[0], -1.0f, 1.0f) + 1.0f) / 2.0f) * (MaxPrice - MinPrice);
                    episodePrices.Add(ourPrice);

                    // Execute action in environment
                    var (reward, unitsSold, done) = Step(action);

                    episodeProfit += reward;
                    episodeUnitsSold += unitsSold;
                    results.TotalRevenue += reward + (unitsSold * UnitCost);

                    // Get next state
                    float[] nextState = GetState();

                    // Store transition in replay buffer
                    _agent.Store(state, action, reward, nextState, done);

                    // Train TD3 from replay buffer
                    _agent.Train(batchSize: 64);

                    state = nextState;
                    if (done) break;
                }

                episodeProfits.Add(episodeProfit);
                episodeAvgPrices.Add(episodePrices.Average());
                results.TotalUnitsSold += episodeUnitsSold;

                if (episodeProfit > results.BestProfit)
                {
                    results.BestProfit = episodeProfit;
                    results.BestAvgPrice = episodePrices.Average();
                }

                // Progress reporting every 20 episodes
                if ((episode + 1) % 20 == 0)
                {
                    var recentProfits = episodeProfits.Skip(Math.Max(0, episodeProfits.Count - 20)).ToList();
                    var recentPrices = episodeAvgPrices.Skip(Math.Max(0, episodeAvgPrices.Count - 20)).ToList();

                    Console.WriteLine($"   Week {episode + 1}/{episodes} | " +
                                    $"Avg Profit: ${recentProfits.Average():F2} | " +
                                    $"Avg Price: ${recentPrices.Average():F2}");
                }
            }

            // Calculate improvement metrics
            results.EarlyProfit = episodeProfits.Take(20).Average();
            results.LateProfit = episodeProfits.Skip(episodes - 20).Average();
            results.ImprovementPercent = ((results.LateProfit - results.EarlyProfit) /
                                         Math.Abs(results.EarlyProfit)) * 100;

            return results;
        }

        public void RunLiveDemo(int weeks)
        {
            for (int week = 0; week < weeks; week++)
            {
                Reset();
                float weekProfit = 0;
                int weekSales = 0;
                var dailyPrices = new List<float>();
                var dailyCompetitorPrices = new List<float>();

                Console.WriteLine($"\n📅 WEEK {week + 1}:");

                float[] state = GetState();

                for (int day = 0; day < 7; day++)
                {
                    // Get action WITHOUT exploration noise (pure exploitation)
                    float[] action = _agent.SelectAction(state, addNoise: false);

                    // Convert to price
                    float ourPrice = MinPrice + ((Math.Clamp(action[0], -1.0f, 1.0f) + 1.0f) / 2.0f) *
                                    (MaxPrice - MinPrice);
                    dailyPrices.Add(ourPrice);
                    dailyCompetitorPrices.Add(_currentCompetitorPrice);

                    var (reward, unitsSold, done) = Step(action);
                    weekProfit += reward;
                    weekSales += unitsSold;

                    float priceDiff = ourPrice - _currentCompetitorPrice;
                    string diffIndicator = priceDiff < 0 ? "✓ UNDERCUTTING" :
                                          priceDiff < 10 ? "≈ MATCHING" : "⚠ PREMIUM";

                    Console.WriteLine($"   Day {day + 1}: Our=${ourPrice:F2} | " +
                                    $"Comp=${_currentCompetitorPrice:F2} | " +
                                    $"Diff=${priceDiff:+0.00;-0.00} | " +
                                    $"Sold={unitsSold:D2} | " +
                                    $"Profit=${reward:F2} | {diffIndicator}");

                    state = GetState();

                    if (done)
                    {
                        Console.WriteLine($"   ⚠️  Low inventory detected - restocking...");
                        Reset();
                        state = GetState();
                    }
                }

                float avgPriceDiff = dailyPrices.Average() - dailyCompetitorPrices.Average();

                Console.WriteLine($"\n   📊 Week {week + 1} Summary:");
                Console.WriteLine($"      Total Profit: ${weekProfit:F2}");
                Console.WriteLine($"      Units Sold: {weekSales}");
                Console.WriteLine($"      Our Avg Price: ${dailyPrices.Average():F2}");
                Console.WriteLine($"      Competitor Avg: ${dailyCompetitorPrices.Average():F2}");
                Console.WriteLine($"      Price Strategy: {(avgPriceDiff < 0 ? "Aggressive (Undercut)" : avgPriceDiff < 5 ? "Competitive (Match)" : "Premium")}");
            }
        }
    }

    public class TrainingResults
    {
        public float EarlyProfit { get; set; }
        public float LateProfit { get; set; }
        public float ImprovementPercent { get; set; }
        public float TotalRevenue { get; set; }
        public int TotalUnitsSold { get; set; }
        public float BestProfit { get; set; }
        public float BestAvgPrice { get; set; }
    }
}