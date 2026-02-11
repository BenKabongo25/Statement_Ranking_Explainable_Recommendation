import ast
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Set


def load_data(csv_path: str, train_path: str, eval_path: str, test_path: str) -> tuple:
    """Load CSV data and split indices."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loading split indices...")
    train_indices = np.load(train_path)
    eval_indices = np.load(eval_path)
    test_indices = np.load(test_path)
    
    print(f"Original dataset size: {len(df)}")
    print(f"Train split size: {len(train_indices)}")
    print(f"Eval split size: {len(eval_indices)}")
    print(f"Test split size: {len(test_indices)}")
    
    return df, train_indices, eval_indices, test_indices


def parse_statement_ids(statement_ids_str: str) -> str:
    """Parse statement_ids from string representation to comma-separated format."""
    try:
        statement_ids = ast.literal_eval(statement_ids_str)
        return ','.join(map(str, statement_ids))
    except:
        return ""


def create_recbole_inter(df: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Create RecBole .inter format dataframe from original data."""
    # Filter dataframe by indices
    filtered_df = df.iloc[indices].copy()
    
    # Create RecBole format
    recbole_df = pd.DataFrame({
        'user_id:token': filtered_df['user_id'].values,
        'item_id:token': filtered_df['item_id'].values,
        'rating:float': filtered_df['rating'].values,
        'tag:token_seq': filtered_df['statement_ids'].apply(parse_statement_ids).values
    })
    
    return recbole_df


def create_recbole_user(user_ids: Set[str]) -> pd.DataFrame:
    """Create RecBole .user format dataframe."""
    return pd.DataFrame({
        'user_id:token': sorted(list(user_ids))
    })


def create_recbole_item(item_ids: Set[str]) -> pd.DataFrame:
    """Create RecBole .item format dataframe."""
    return pd.DataFrame({
        'item_id:token': sorted(list(item_ids))
    })


def save_recbole_files(
    output_dir: Path,
    dataset_name: str,
    all_inter: pd.DataFrame,
    train_inter: pd.DataFrame,
    eval_inter: pd.DataFrame,
    test_inter: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame
):
    """Save all RecBole format files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save .inter files
    print(f"\nSaving interaction files...")
    all_inter.to_csv(
        output_dir / f"{dataset_name}.inter",
        sep='\t',
        index=False
    )
    print(f"Saved {dataset_name}.inter ({len(all_inter)} interactions)")
    
    train_inter.to_csv(
        output_dir / f"{dataset_name}.train.inter",
        sep='\t',
        index=False
    )
    print(f"Saved {dataset_name}.train.inter ({len(train_inter)} interactions)")
    
    eval_inter.to_csv(
        output_dir / f"{dataset_name}.eval.inter",
        sep='\t',
        index=False
    )
    print(f"Saved {dataset_name}.eval.inter ({len(eval_inter)} interactions)")
    
    test_inter.to_csv(
        output_dir / f"{dataset_name}.test.inter",
        sep='\t',
        index=False
    )
    print(f"Saved {dataset_name}.test.inter ({len(test_inter)} interactions)")
    
    # Save .user file
    user_df.to_csv(
        output_dir / f"{dataset_name}.user",
        sep='\t',
        index=False
    )
    print(f"\nSaved {dataset_name}.user ({len(user_df)} users)")
    
    # Save .item file
    item_df.to_csv(
        output_dir / f"{dataset_name}.item",
        sep='\t',
        index=False
    )
    print(f"Saved {dataset_name}.item ({len(item_df)} items)")


def main(args):
    # Load data
    df, train_indices, eval_indices, test_indices = load_data(
        args.input_csv,
        args.train_split,
        args.eval_split,
        args.test_split
    )
    
    # Get all retained indices
    all_indices = np.concatenate([train_indices, eval_indices, test_indices])
    
    # Create interaction dataframes for each split
    print("\nCreating RecBole format dataframes...")
    all_inter = create_recbole_inter(df, all_indices)
    train_inter = create_recbole_inter(df, train_indices)
    eval_inter = create_recbole_inter(df, eval_indices)
    test_inter = create_recbole_inter(df, test_indices)
    
    # Extract unique users and items from all retained interactions
    unique_users = set(all_inter['user_id:token'].unique())
    unique_items = set(all_inter['item_id:token'].unique())
    
    print(f"\nTotal unique users: {len(unique_users)}")
    print(f"Total unique items: {len(unique_items)}")
    
    # Create user and item dataframes
    user_df = create_recbole_user(unique_users)
    item_df = create_recbole_item(unique_items)
    
    # Save all files
    output_dir = Path(args.output_dir)
    save_recbole_files(
        output_dir,
        args.dataset_name,
        all_inter,
        train_inter,
        eval_inter,
        test_inter,
        user_df,
        item_df
    )
    
    print("\n✓ Conversion completed successfully!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (will be used as prefix for output files)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        required=True,
        help="Path to train split indices (.npy file)"
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        required=True,
        help="Path to eval split indices (.npy file)"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        required=True,
        help="Path to test split indices (.npy file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where RecBole format files will be saved"
    )
    
    args = parser.parse_args()
    main(args)
