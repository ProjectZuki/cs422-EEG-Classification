# CS422-EEG-Classification

CS422 UNLV Spring '24 | Electroencephalography (EEG) Classification to detect harmful brain activity.

This project is the collaborative work of the following individuals

## [ProjectZuki](https://github.com/ProjectZuki) | [arianizadi](https://github.com/arianizadi) | [fourfourfourfourthreethreethree](https://github.com/fourfourfourfourthreethreethree)

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Dependencies](#dependencies)
5. [License](#license)

## Overview

This project aims to apply machine learning techniques to classify and detect harmful brain activity from electroencephalography (EEG) data. The goal is to detect various types of abnormal brain patterns, including seizures and other detrimental activities, to aid in medical diagnostics.

## Dataset

Datasets are provided by Harvard Medical School. Details as well as the dataset can be obtained [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data).
Or by running the command under [Dependencies](#dependencies)

## Usage
`test.py`: 
```bash
python test.py
```

`unittest.py`
```bash
python unittest.py
```

## Dependencies
When running this project, the following python libraries are required:
- pandas
- numpy
- tensorflow
- sklearn

```bash
pip install pandas numpy tensorflow scikit-learn pyarrow
```

Download DataSet:
```bash
kaggle competitions download -c hms-harmful-brain-activity-classification
```

## License

This project is licensed under the [MIT License](LICENSE), allowing users to freely use, modify, and distribute the code with proper attribution.
