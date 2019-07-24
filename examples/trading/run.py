from examples.trading.mlp_trading import MlpTrading

mlp_trading = MlpTrading(symbol='^GSPC')
mlp_trading.execute(skip_days=3600,
                    epochs=500,
                    size_hidden=512,
                    batch_size=128,
                    percent_test_split=0.33)
