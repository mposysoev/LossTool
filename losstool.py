import numpy as np
import argparse
import sys
from typing import Dict, Tuple

def load_data(filename: str) -> np.ndarray:
    """
    Load data from a two-column file.
    
    Args:
        filename (str): Path to the input file.
    
    Returns:
        np.ndarray: Loaded data with shape (2, n).
    
    Raises:
        ValueError: If the file doesn't contain exactly two columns.
    """
    try:
        data = np.genfromtxt(filename).T
        if data.shape[0] != 2:
            raise ValueError(f"Input file {filename} must contain exactly two columns.")
        return data
    except IOError:
        print(f"Error: Unable to read file '{filename}'. Please check if the file exists and you have read permissions.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_losses(ref_data: np.ndarray, obtained_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate different loss functions.
    
    Args:
        ref_data (np.ndarray): Reference data with shape (2, n).
        obtained_data (np.ndarray): Obtained data with shape (2, n).
    
    Returns:
        Dict[str, float]: Dictionary containing calculated loss values.
    """
    y_true = ref_data[1, :]
    y_pred = obtained_data[1, :]
    
    losses = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R-squared": r_squared(y_true, y_pred)
    }
    return losses

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate loss values between reference and obtained data.")
    parser.add_argument("ref_file", help="Path to the reference data file")
    parser.add_argument("obtained_file", help="Path to the obtained data file")
    parser.add_argument("-p", "--precision", type=int, default=10, help="Number of decimal places for output (default: 10)")
    return parser.parse_args()

def check_data_consistency(ref_data: np.ndarray, obtained_data: np.ndarray) -> None:
    """
    Check if both datasets have the same shape.
    
    Raises:
        ValueError: If the datasets have different shapes.
    """
    if ref_data.shape != obtained_data.shape:
        raise ValueError("Reference and obtained data must have the same shape.")

def main() -> None:
    args = parse_arguments()
    
    ref_data = load_data(args.ref_file)
    obtained_data = load_data(args.obtained_file)
    
    try:
        check_data_consistency(ref_data, obtained_data)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    losses = calculate_losses(ref_data, obtained_data)
    
    print("Loss values:")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value:.{args.precision}f}")
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print(f"Number of data points: {ref_data.shape[1]}")
    print(f"X range: {ref_data[0, :].min():.{args.precision}f} to {ref_data[0, :].max():.{args.precision}f}")
    print(f"Y range (reference): {ref_data[1, :].min():.{args.precision}f} to {ref_data[1, :].max():.{args.precision}f}")
    print(f"Y range (obtained): {obtained_data[1, :].min():.{args.precision}f} to {obtained_data[1, :].max():.{args.precision}f}")

if __name__ == "__main__":
    main()
