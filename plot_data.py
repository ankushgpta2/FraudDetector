import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotting():
    def __init__(self):
        # initialize utility class 
        self.PlottingUtility = PlottingUtility()
        # initialize some values
        self.optimal_bin_width = 0
        self.num_bins = 0

    def plot_histogram(self, data):
        """
        """
        # make sure that the data are numbers 
        try:
            data = data.apply(PlottingUtility.ensure_numeric)
        except ValueError as e:
            print(e)
        
        # get optimal bindwidth for data 
        self.optimal_bin_width = self.PlottingUtility.calculate_optimal_bindwidth(data)

        # calculate # of bins based on bin width
        self.num_bins = int((max(data) - min(data)) / self.optimal_bin_width)

        # utilize bindwidth to create histogram for data
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=self.num_bins, edgecolor='black', color='skyblue', alpha=0.7)

        # add grid lines
        plt.grid(True, linestyle='--', alpha=0.6)

        # add summary statistics lines
        mean_value = np.mean(data)
        median_value = np.median(data)
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
        plt.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value:.2f}')

        # add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.legend()

        # show the plot
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, correlation_matrix):
        """
        """
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.show()


class PlottingUtility():
    def __init__(self):
        """
        """

    @staticmethod
    def calculate_optimal_bindwidth(data):
        """
        """
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        bin_width = 2 * iqr * len(data) ** (-1 / 3)
        return bin_width

    @staticmethod
    def ensure_numeric(value):
        try:
            # try to convert the value to a float
            return float(value)
        except ValueError:
            # if conversion fails, raise an error
            raise ValueError(f"Cannot Convert {value} to A Number When Plotting Histogram.")
