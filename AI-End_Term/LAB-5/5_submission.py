import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


class StockHMMAnalyzer:
    def __init__(
        self, ticker, start_date="2013-01-01", end_date="2023-01-01", n_states=3
    ):
        
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = n_states

        self.raw_data = None
        self.processed_data = None
        self.model = None

    def fetch_data(self):
       self.raw_data = yf.download(
            self.ticker, start=self.start_date, end=self.end_date
        )

        self.raw_data.to_csv(f"{self.ticker}_stock_data.csv")

        return self.raw_data

    def preprocess_data(self):
       
        self.processed_data = self.raw_data[["Adj Close"]].copy()
        self.processed_data["Returns"] = self.processed_data["Adj Close"].pct_change()

        self.processed_data["Log Returns"] = np.log(1 + self.processed_data["Returns"])
        self.processed_data["Rolling Volatility"] = (
            self.processed_data["Returns"].rolling(window=20).std()
        )

        self.processed_data.dropna(inplace=True)

        return self.processed_data

    def fit_hmm(self):
        

        features = self.processed_data[["Log Returns", "Rolling Volatility"]].values

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        self.model.fit(scaled_features)

        hidden_states = self.model.predict(scaled_features)
        self.processed_data["Hidden State"] = hidden_states

        return hidden_states

    def analyze_hmm_parameters(self):
       
        print("HMM Model Parameters:")
        print("\nMeans:")
        print(self.model.means_)

        print("\nCovariances:")
        print(self.model.covars_)

        print("\nTransition Matrix:")
        print(self.model.transmat_)

        return {
            "means": self.model.means_,
            "covariances": self.model.covars_,
            "transition_matrix": self.model.transmat_,
        }

    def visualize_states(self):
        
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.title(f"{self.ticker} Stock Price with Hidden States")
        plt.plot(
            self.processed_data.index,
            self.processed_data["Adj Close"],
            label="Adjusted Close",
            color="black",
        )

        scatter = plt.scatter(
            self.processed_data.index,
            self.processed_data["Adj Close"],
            c=self.processed_data["Hidden State"],
            cmap="viridis",
            marker=".",
        )
        plt.colorbar(scatter, label="Hidden State")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")

        plt.subplot(2, 1, 2)
        plt.title("Log Returns by Hidden State")
        for state in range(self.n_states):
            state_returns = self.processed_data[
                self.processed_data["Hidden State"] == state
            ]["Log Returns"]
            plt.hist(
                state_returns, bins=50, alpha=0.5, label=f"State {state}", density=True
            )

        plt.xlabel("Log Returns")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict_next_state(self):

        current_state = self.processed_data["Hidden State"].iloc[-1]
        next_state_probabilities = self.model.transmat_[current_state]

        print(f"\nCurrent State: {current_state}")
        print("Next State Probabilities:")
        for state, prob in enumerate(next_state_probabilities):
            print(f"State {state}: {prob:.4f}")

        next_state = np.argmax(next_state_probabilities)
        print(f"\nMost Likely Next State: {next_state}")

        return next_state, next_state_probabilities

    def run_analysis(self):

        self.fetch_data()
        self.preprocess_data()
        se
        lf.fit_hmm()
        self.analyze_hmm_parameters()
        self.visualize_states()
        self.predict_next_state()


def main():

    msft_analyzer = StockHMMAnalyzer("MSFT", n_states=3)
    msft_analyzer.run_analysis()


if __name__ == "__main__":
    main()
