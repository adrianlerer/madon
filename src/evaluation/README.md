# Evaluation Script Documentation

This comprehensive documentation covers all features and use cases of the evaluate_all.py script for classification model evaluation.

## Overview

The evaluate_all.py script is a powerful tool for evaluating binary and multilabel classification results. It supports three different configuration modes, statistical analysis, confusion matrix generation, and flexible output formatting.

## Table of Contents

- Basic Usage
- Configuration Modes
- Command Line Arguments
- Input File Format
- Configuration Details
- Advanced Features
- Output Formats
- Edge Cases and Validation
- Examples
- Troubleshooting

## Basic Usage

```bash
python evaluate_all.py [CSV_FILES...] --config [1|2|3] [OPTIONS]
```

### Minimum Required Arguments
- `CSV_FILES`: One or more CSV files containing classification results
- `--config`: Configuration mode (1, 2, or 3)

## Configuration Modes

### Config 1: Binary Classification (Basic)
- **Purpose**: Evaluate binary classification with basic metrics
- **Input**: Gold and Predicted labels as 0/1 values
- **Metrics**: Macro Precision, Macro Recall, Macro F1
- **Use Case**: Simple binary classification tasks

### Config 2: Binary Classification (Extended)
- **Purpose**: Same as Config 1 but with additional features
- **Input**: Gold and Predicted labels as 0/1 values
- **Metrics**: Macro Precision, Macro Recall, Macro F1
- **Additional Features**: Confusion matrix generation support
- **Use Case**: Binary classification with visualization needs

### Config 3: Multilabel Classification
- **Purpose**: Evaluate multilabel classification tasks
- **Input**: Gold and Predicted labels as 8-bit binary strings (e.g., "10110000")
- **Labels**: 8 predefined labels: ["LIN", "SI", "CL", "D", "HI", "PL", "TI", "PC"]
- **Metrics**: F1 scores (positive, negative, macro) for each label
- **Use Case**: Complex multilabel classification tasks

## Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `csv_files` | list | One or more CSV files to evaluate |
| `--config` | int | Configuration mode: 1, 2, or 3 |

### Optional Arguments

| Argument | Availability | Description |
|----------|--------------|-------------|
| `--out_csv` | All configs | Save results to CSV file |
| `--std_dev` | All configs | Calculate and include standard deviation |
| `--matrix` | Config 1, 2 only | Generate confusion matrix plots |
| `--pre_rec` | Config 3 only | Include precision and recall metrics |
| `--confusion_3` | Config 3 only | Generate aggregated confusion matrix |

## Input File Format

### For Config 1 & 2 (Binary Classification)

```csv
Gold Labels,Predicted Labels
0,0
1,1
0,1
1,0
```

**Requirements:**
- Two columns: "Gold Labels" and "Predicted Labels"
- Values must be 0 or 1
- No missing values allowed

### For Config 3 (Multilabel Classification)

```csv
Gold Labels,Predicted Labels
"10110000","10010000"
"[1, 0, 1, 1, 0, 0, 0, 0]","[1, 0, 0, 1, 0, 0, 0, 0]"
"01000001","01000001"
```

**Supported Formats:**
1. **String format**: `"10110000"` (8 digits, each 0 or 1)
2. **List format**: `"[1, 0, 1, 1, 0, 0, 0, 0]"` (8 elements, each 0 or 1)

**Requirements:**
- Two columns: "Gold Labels" and "Predicted Labels"
- Exactly 8 binary values per label
- Consistent format within each file
- No missing values allowed

## Configuration Details

### Config 1 - Basic Binary Classification

**Features:**
- Basic binary classification metrics
- Support for multiple files with averaging
- Optional standard deviation calculation
- Optional CSV export

**Limitations:**
- No confusion matrix support
- No visualization features

**Example:**
```bash
python evaluate_all.py data1.csv data2.csv --config 1
python evaluate_all.py data1.csv --config 1 --std_dev --out_csv results.csv
```

### Config 2 - Extended Binary Classification

**Features:**
- All Config 1 features
- Confusion matrix generation with `--matrix` flag
- Enhanced visualization capabilities

**Matrix Generation:**
- Creates PNG files in `appendix/` directory
- Naming pattern: `{filename}_confusion_matrix.png`
- 4x4 plots with color-coded confusion matrices

**Example:**
```bash
python evaluate_all.py data1.csv data2.csv --config 2 --matrix
python evaluate_all.py data1.csv --config 2 --matrix --std_dev --out_csv results.csv
```

### Config 3 - Multilabel Classification

**Features:**
- Multilabel classification support
- Per-label and macro-average metrics
- Optional precision/recall with `--pre_rec`
- Optional aggregated confusion matrix with `--confusion_3`
- Flexible label format handling

**Label Processing:**
- Supports both string and list formats
- Automatic format detection and normalization
- Handles different bracket styles and spacing

**Metrics Calculated:**
- **Standard mode**: Positive F1, Negative F1, Macro F1
- **With --pre_rec**: Additionally includes Precision and Recall for each category

**Example:**
```bash
python evaluate_all.py data1.csv data2.csv --config 3
python evaluate_all.py data1.csv --config 3 --pre_rec --std_dev
python evaluate_all.py data1.csv --config 3 --confusion_3 --out_csv results.csv
```

## Advanced Features

### Standard Deviation Calculation (`--std_dev`)

**Purpose:** Calculate and report standard deviation across multiple input files

**Availability:** All configurations

**Output Changes:**
- Console: Metrics displayed as `Mean ± StdDev`
- CSV: Additional standard deviation columns added

**Requirements:**
- Multiple input files (>1) for meaningful calculation
- Single file results in 0.0 standard deviation

**Example:**
```bash
# With standard deviation
python evaluate_all.py file1.csv file2.csv file3.csv --config 1 --std_dev

# Output format: "Macro Precision: 85.23 ± 2.15"
```

### Precision and Recall Analysis (`--pre_rec`)

**Purpose:** Include detailed precision and recall metrics for multilabel tasks

**Availability:** Config 3 only

**Additional Metrics:**
- Positive Precision, Positive Recall
- Negative Precision, Negative Recall  
- Macro Precision, Macro Recall
- All metrics calculated per label and averaged

**CSV Output Changes:**
- 9 rows per experiment instead of 3
- Separate rows for F1, Precision, and Recall
- Additional standard deviation columns when combined with `--std_dev`

**Example:**
```bash
python evaluate_all.py data.csv --config 3 --pre_rec --std_dev --out_csv detailed_results.csv
```

### Confusion Matrix Generation

#### Binary Classification (`--matrix`)

**Availability:** Config 1 and 2

**Features:**
- Individual matrix per input file
- 2x2 confusion matrix (TP, FP, FN, TN)
- Saved as high-resolution PNG files
- Color-coded visualization

**File Output:**
- Directory: `appendix/`
- Naming: `{filename}_confusion_matrix.png`
- Format: PNG, 300 DPI

#### Multilabel Classification (`--confusion_3`)

**Availability:** Config 3 only

**Features:**
- Aggregated confusion matrix across all labels
- Sums TP, FP, FN, TN from all 8 labels
- Enhanced visualization with metrics overlay
- Console output of aggregated statistics

**Metrics Overlay:**
- Overall Accuracy
- Overall Precision
- Overall Recall
- Overall F1-Score

**Example:**
```bash
python evaluate_all.py data1.csv data2.csv --config 3 --confusion_3
```

## Output Formats

### Console Output

#### Basic Metrics (Config 1 & 2)
```
[Config 1 - Binary (File 1)]
Macro Precision:          0.8523
Macro Recall:             0.7891  
Macro F1:                 0.8193

=== Average Binary Metrics ===
Macro Precision: 85.23 ± 2.15
Macro Recall: 78.91 ± 3.42
Macro F1: 81.93 ± 1.87
```

#### Multilabel Metrics (Config 3)
```
=== Average Multilabel Metrics ===
LIN | Pos F1: 78.45 | Neg F1: 92.31 | Macro F1: 85.38
SI  | Pos F1: 65.23 | Neg F1: 88.76 | Macro F1: 76.99
...
Macro Avg | Pos F1: 71.23 | Neg F1: 89.45 | Macro F1: 80.34
```

#### With Precision/Recall (Config 3 + --pre_rec)
```
LIN | Pos F1: 78.45±2.1 | Neg F1: 92.31±1.5 | Macro F1: 85.38±1.2
    | Pos Prec: 76.23±2.3 | Neg Prec: 94.12±1.1 | Macro Prec: 85.17±1.4
    | Pos Rec: 80.78±3.1 | Neg Rec: 90.56±2.0 | Macro Rec: 85.67±1.8
```

### CSV Output

#### Binary Classification CSV Structure

**Without Standard Deviation:**
```csv
Experiment,Macro Precision,Macro Recall,Macro F1
experiment1,85.23,78.91,81.93
```

**With Standard Deviation:**
```csv
Experiment,Macro Precision,Macro Recall,Macro F1,Macro Precision Std,Macro Recall Std,Macro F1 Std
experiment1,85.23,78.91,81.93,2.15,3.42,1.87
```

#### Multilabel Classification CSV Structure

**Standard Format:**
```csv
Experiment,Type,LIN,SI,CL,D,HI,PL,TI,PC,Macro Avg
experiment1,Positive,78.45,65.23,82.11,73.56,69.78,88.92,75.34,71.23,75.58
experiment1,Negative,92.31,88.76,94.22,89.67,91.45,96.78,90.12,89.45,91.60
experiment1,Average,85.38,76.99,88.16,81.61,80.61,92.85,82.73,80.34,83.59
```

**With Precision/Recall:**
```csv
Experiment,Type,LIN,SI,CL,D,HI,PL,TI,PC,Macro Avg
experiment1,Positive_F1,78.45,65.23,...
experiment1,Negative_F1,92.31,88.76,...
experiment1,Average_F1,85.38,76.99,...
experiment1,Positive_Precision,76.23,63.45,...
experiment1,Negative_Precision,94.12,90.23,...
experiment1,Average_Precision,85.17,76.84,...
experiment1,Positive_Recall,80.78,67.12,...
experiment1,Negative_Recall,90.56,87.34,...
experiment1,Average_Recall,85.67,77.23,...
```

## Edge Cases and Validation

### Input Validation

1. **File Existence Check**
   - Script validates all CSV files exist before processing
   - Clear error message if files are missing

2. **Column Validation**
   - Ensures required columns are present
   - "Gold Labels" and "Predicted Labels" must exist

3. **Data Type Validation**
   - Binary configs: Values must be 0 or 1
   - Multilabel config: Labels must be 8-bit binary strings

4. **Label Length Validation (Config 3)**
   - Exactly 8 binary values required
   - Error thrown for incorrect lengths

### Argument Validation

1. **Flag Compatibility Checks**
   ```bash
   # These will produce errors:
   python evaluate_all.py data.csv --config 1 --pre_rec          # Error: --pre_rec only for config 3
   python evaluate_all.py data.csv --config 3 --matrix           # Error: --matrix only for config 1,2
   python evaluate_all.py data.csv --config 1 --confusion_3      # Error: --confusion_3 only for config 3
   ```

2. **File Count Validation**
   - At least one input file required
   - Standard deviation meaningful only with multiple files

### Data Processing Edge Cases

1. **Division by Zero Protection**
   - All metric calculations handle zero denominators
   - Returns 0.0 for undefined metrics (e.g., precision when TP+FP=0)

2. **Empty Files**
   - Script handles empty CSV files gracefully
   - Reports zero metrics for empty datasets

3. **Mixed Label Formats (Config 3)**
   - Automatic detection and normalization
   - Supports mixing of string and list formats within same file

4. **Special Cases in Multilabel**
   - "No argument" detection (all zeros: "00000000")
   - Perfect match counting
   - Hamming distance calculation

## Examples

### Basic Usage Examples

1. **Simple Binary Evaluation**
   ```bash
   python evaluate_all.py results.csv --config 1
   ```

2. **Multiple Files with Averaging**
   ```bash
   python evaluate_all.py results1.csv results2.csv results3.csv --config 2 --std_dev
   ```

3. **Multilabel with All Features**
   ```bash
   python evaluate_all.py multilabel_results.csv --config 3 --pre_rec --confusion_3 --std_dev
   ```

### CSV Export Examples

1. **Binary Results to CSV**
   ```bash
   python evaluate_all.py binary1.csv binary2.csv --config 2 --matrix --std_dev --out_csv binary_results.csv
   ```

2. **Comprehensive Multilabel Analysis**
   ```bash
   python evaluate_all.py ml1.csv ml2.csv ml3.csv --config 3 --pre_rec --std_dev --out_csv detailed_multilabel.csv
   ```

### Visualization Examples

1. **Binary Confusion Matrices**
   ```bash
   python evaluate_all.py test1.csv test2.csv --config 2 --matrix
   # Creates: appendix/test1_confusion_matrix.png, appendix/test2_confusion_matrix.png
   ```

2. **Multilabel Aggregated Matrix**
   ```bash
   python evaluate_all.py multilabel.csv --config 3 --confusion_3
   # Creates: appendix/multilabel_multilabel_confusion_matrix.png
   ```

### Complex Workflow Examples

1. **Complete Binary Analysis Pipeline**
   ```bash
   # Step 1: Generate matrices and basic metrics
   python evaluate_all.py exp1_seed1.csv exp1_seed2.csv exp1_seed3.csv --config 2 --matrix --std_dev --out_csv exp1_results.csv
   
   # Step 2: Combine with other experiments
   python evaluate_all.py exp2_seed1.csv exp2_seed2.csv exp2_seed3.csv --config 2 --std_dev --out_csv exp1_results.csv
   ```

2. **Comprehensive Multilabel Study**
   ```bash
   # Full analysis with all features
   python evaluate_all.py model_a_*.csv --config 3 --pre_rec --confusion_3 --std_dev --out_csv comprehensive_results.csv
   ```

## Troubleshooting

### Common Errors and Solutions

1. **"FileNotFoundError"**
   - **Cause**: CSV file doesn't exist
   - **Solution**: Check file paths and ensure files exist

2. **"KeyError: 'Gold Labels'"**
   - **Cause**: Missing required columns
   - **Solution**: Ensure CSV has "Gold Labels" and "Predicted Labels" columns

3. **"ValueError: Unsupported label format"**
   - **Cause**: Invalid label format in multilabel data
   - **Solution**: Ensure labels are 8-bit binary strings or lists

4. **"--pre_rec is only available for config 3"**
   - **Cause**: Using precision/recall flag with binary configs
   - **Solution**: Use --pre_rec only with --config 3

5. **"Row X has invalid label lengths"**
   - **Cause**: Multilabel data doesn't have exactly 8 values
   - **Solution**: Check data format and ensure 8 binary values per label

### Performance Considerations

1. **Large Files**: Script processes files sequentially; large files may take time
2. **Multiple Files**: Processing time scales linearly with number of files
3. **Matrix Generation**: PNG creation adds processing time but improves analysis

### Best Practices

1. **File Naming**: Use descriptive names that will create meaningful experiment names
2. **Data Validation**: Validate your CSV format before running the script
3. **Output Organization**: Use consistent output CSV names to avoid conflicts
4. **Flag Combination**: Combine compatible flags for comprehensive analysis
5. **Directory Structure**: Ensure write permissions for appendix/ directory creation

This documentation covers all aspects of the evaluation script. For additional questions or edge cases not covered here, refer to the script's built-in help:

```bash
python evaluate_all.py --help
```