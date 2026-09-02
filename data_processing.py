
from csv import writer

import pandas as pd
import numpy as np


class DataProcessor:
    """
    *test diff*
    A class to handle data processing, including importing data, applying masking logic,
    and saving processed data to Excel files.
    """
    
    def __init__(
            self,
            excel_file_path,
        ):
        """Initialize the DataProcessor with raw data from an Excel file."""
        self.excel_file_path = excel_file_path
        self.raw_data = None
        self.data_dict = {}
        self.solution_pool = set()
        self.indexing_dict = {}
        self.inverse_indexing_dict = {}
        self.comb_mat = None
        
        # Masking attributes
        self.isolation_mask_tracking = None
        self.comb_mask_list = []
        self.sol_mask_list = []
        self.remaining_comb = []
        self.remaining_sol = []
        self.masked_indexing_dict = {}
        self.masked_inv_indexing_dict = {}
        self.comb_mat_masked = None
        
        # Score data
        self.Y = None
        self.Y_masked = None
        self.Y_masked_labeled = None
        
        # Label dictionaries
        self.label_dict = {
            -2: 'significantly ineffective',
            -1: 'ineffective',
            0: 'neutral',
            1: 'effective',
            2: 'significantly effective'
        }
    
    def import_data(self):
        """Import and preprocess raw data from Excel file."""
        self.raw_data = pd.read_excel(self.excel_file_path)
        self.raw_data = self.raw_data.drop(columns=['config.'])
        self.raw_data['classification by result'] = self.raw_data['classification by result'].apply(lambda x: x.strip())
        self.raw_data['Score'] = self.raw_data['classification by result'].map(
            {
                'significantly effective': 2,
                'effective': 1,
                'mix': 0,
                'MIX': 0,
                'ineffective': -1,
                'significantly ineffective': -2
            }
        )

    def build_combination_matrix(self):
        """Build data dictionary, solution pool, and combination matrix."""
        # Build data_dict and solution_pool
        for comb in range(self.raw_data.shape[0]):
            sol_list = [x.strip() for x in [str(x) for x in self.raw_data.iloc[comb][:-2].to_list()] if x != 'nan']
            self.data_dict[comb] = {
                'solutions': sol_list,
                'score': self.raw_data.iloc[comb, -1]
            }
            self.solution_pool.update(sol_list)
        
        # Build indexing dictionaries
        self.indexing_dict = {s: i for i, s in enumerate(self.solution_pool)}
        self.inverse_indexing_dict = {i: s for s, i in self.indexing_dict.items()}
        
        # Build combination matrix
        self.comb_mat = np.empty((len(self.data_dict), len(self.solution_pool)))
        for comb in self.data_dict:
            for sol in self.data_dict[comb]['solutions']:
                self.comb_mat[comb, self.indexing_dict[sol]] = 1
            self.comb_mat[comb, np.isnan(self.comb_mat[comb])] = 0
    
    def apply_isolation_mask(self):
        """
        Apply masking to remove solutions and combinations that only appear in isolation.
        Masks combinations that contain only singular solutions (solutions appearing in only one combination).
        """
        # Identify combinations with singular solutions
        self.isolation_mask_tracking = pd.DataFrame(
            data={
                "suspected combination index": np.where(self.comb_mat[:, (self.comb_mat.sum(axis=0) == 1)].sum(axis=1) > 0)[0]
            }
        )
        
        # Identify singular solutions and combinations containing only them
        single_comb_bool = (self.comb_mat.sum(axis=0) == 1)
        single_comb_participation = self.comb_mat[self.comb_mat[:, single_comb_bool].sum(axis=1) > 0, :]
        combinations_w_singular_solutions = (single_comb_participation.sum(axis=1) - single_comb_participation[:, single_comb_bool].sum(axis=1)) == 0
        
        self.isolation_mask_tracking["isolated combination"] = combinations_w_singular_solutions.astype(int)
        self.comb_mask_list = self.isolation_mask_tracking[self.isolation_mask_tracking["isolated combination"] == 1]["suspected combination index"].to_list()
        
        # Track solution and combination lists
        self.isolation_mask_tracking["solution list"] = self.isolation_mask_tracking["suspected combination index"].apply(lambda x: self.data_dict[x]['solutions'])
        self.isolation_mask_tracking["solution index list"] = self.isolation_mask_tracking["solution list"].apply(lambda x: [self.indexing_dict[sol] for sol in x])
        
        # Remaining combinations after masking singular combinations
        self.remaining_comb = [i for i in range(self.comb_mat.shape[0]) if i not in self.comb_mask_list]
        
        # Remaining solutions after masking singular solutions
        self.sol_mask_list = []
        for comb in self.isolation_mask_tracking[self.isolation_mask_tracking["isolated combination"] == 1]["solution index list"].to_list():
            for sol in comb:
                if sol not in self.sol_mask_list:
                    self.sol_mask_list.append(sol)
        self.remaining_sol = [i for i in range(self.comb_mat.shape[1]) if i not in self.sol_mask_list]
        
        # Create masked indexing dictionaries
        self.masked_indexing_dict = {sol: i for i, sol in enumerate(self.remaining_sol)}
        self.masked_inv_indexing_dict = {i: sol for sol, i in self.masked_indexing_dict.items()}
        
        # Create masked combination matrix
        self.comb_mat_masked = self.comb_mat[self.remaining_comb, :][:, self.remaining_sol]
        
        # Create masked score arrays
        self.Y = np.array([self.data_dict[comb]['score'] for comb in self.data_dict])
        self.Y_masked = np.array([self.data_dict[comb]['score'] for comb in self.data_dict.keys() if comb in self.remaining_comb])
        self.Y_masked_labeled = np.array([self.label_dict[score] for score in self.Y_masked])
    
    def update_labels_and_scores(self, conversion_instructions):
        """
        Update labels and scores according to the provided conversion instructions.
        
        This method updates Y, Y_masked, Y_masked_labeled, label_dict, and rev_label_dict
        without modifying raw_data.
        
        Args:
            conversion_instructions (dict): A dictionary with the following structure:
                {
                    'label': {old_label: new_label, ...},  # Maps old label strings to new ones
                    'score': {old_score: new_score, ...},  # Maps old numeric scores to new ones
                    'updated_label_dict': {score: label, ...}  # New complete label dictionary
                }
                
                Where:
                - 'label' key (required): Dictionary mapping old label strings to new label strings.
                  E.g., {'ineffective': 'neutral', 'effective': 'significantly effective'}
                - 'score' key (required): Dictionary mapping old numeric scores to new numeric scores.
                  E.g., {-1: 0, 1: 2}
                - 'updated_label_dict' key (required): Complete updated mapping of scores to labels.
                  E.g., {-2: 'significantly ineffective', 0: 'neutral', 2: 'significantly effective'}
        """
        
        # Update Y_masked_labeled with label conversions
        if 'label' in conversion_instructions:
            for old_val, new_val in conversion_instructions['label'].items():
                if old_val in self.Y_masked_labeled:
                    self.Y_masked_labeled[self.Y_masked_labeled == old_val] = new_val
        
        # Update Y_masked and Y with score conversions
        if 'score' in conversion_instructions:
            for old_val, new_val in conversion_instructions['score'].items():
                if old_val in self.Y_masked:
                    self.Y_masked[self.Y_masked == old_val] = new_val
                if old_val in self.Y:
                    self.Y[self.Y == old_val] = new_val
        
        # Update label dictionaries
        if 'updated_label_dict' in conversion_instructions:
            self.label_dict = conversion_instructions['updated_label_dict']
    
    def save_to_excel(self, output_prefix='logreg1'):
        """Save all processed data to Excel files."""
        writer = pd.ExcelWriter(f'{output_prefix}_train_data.xlsx', engine='xlsxwriter')
        
        # Training data with combination matrix
        col_names = ["combination index", "label"]
        col_names.extend([self.inverse_indexing_dict[x] for x in self.remaining_sol])
        
        df_data = np.concatenate(
            [
                np.array(self.remaining_comb).reshape((-1, 1)),
                self.Y_masked_labeled.reshape((-1, 1)),
                self.comb_mat_masked
            ],
            axis=1
        )
        train_df = pd.DataFrame(columns=col_names, data=df_data)
        train_df.to_excel(
            writer,
            sheet_name=f'train_data'
            )
        
        # Raw data
        self.raw_data.to_excel(
            writer,
            sheet_name=f'raw_data'
        )
        
        # Solution indexing
        pd.DataFrame(
            data={k: [v] for k, v in self.indexing_dict.items()}
        ).to_excel(
            writer,
            sheet_name=f'solution_indexing'
        )
        
        # Solution mask list
        pd.DataFrame(
            data={self.inverse_indexing_dict[x]: [x] for x in self.sol_mask_list}
        ).to_excel(
            writer,
            sheet_name=f'solution_mask_list'
        )
        
        # Combination mask tracking
        self.isolation_mask_tracking.to_excel(
            writer,
            sheet_name=f'combination_mask_tracking'
        )
        
        writer.close()
    
    def process_all(self):
        """Execute the complete data processing pipeline."""
        self.import_data()
        self.build_combination_matrix()
        self.apply_isolation_mask()


# Initialize and run data processor
if __name__ == "__main__":
    data_processor = DataProcessor('Shiran- raw data for analysis.xlsx')
    data_processor.process_all()
    data_processor.save_to_excel()

