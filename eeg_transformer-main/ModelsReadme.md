# VS Code Setup

Create virtual environment  
`python -m venv venv`

Activate virtual environment  
`.\venv\Scripts\Activate`

Make sure that pip installation points to virtual environment  
`pip --version`

Install dependencies  
`pip install -r requirements.txt`

Install recommended extensions specified in  
`extensions.json`

# Scripts

### **download.py**

This script downloads raw datasets related to motor imagery. It accepts dataset name as a starting param.  
If none is provided, it will download all datasets. Supported dataset names are:  
`bci3a` - for BCI Competition III 3a  
`bci2a` - for BCI Competition IV 2a  
`bci2b` - for BCI Competition IV 2b  
`physionet` - for Physionet

In ```notebook``` folder there is a file called ```eeg_transformer_models_version_final``` which contains the whole code past preprocessing used in the thesis
***
# Results:
**Variant 1**
| Model                  | Metoda | Współczynnik Uczenia | Średnia Dokładność (%) |
| :--------------------- | :----- | :------------------- | :--------------------- |
| SpatialTransformer     | raw    | 0.0001               | 65.92                  |
| SpatialTransformer     | raw    | 0.0007               | 62.68                  |
| TemporalTransformer    | raw    | 0.0007               | 74.85                  |
| TemporalTransformer    | raw    | 0.0001               | 74.42                  |
| SpatialCNNTransformer  | raw    | 0.0007               | 73.42                  |
| TemporalCNNTransformer | raw    | 0.0007               | 73.76                  |
| FusionCNNTransformer   | raw    | 0.0007               | 74.06                  |

**Variant 2**
| Model                  | Metoda | WU     | Liczba punktów FFT | Skok Ramki | Średnia Dokładność (%) |
| :--------------------- | :----- | :----- | :----------------- | :--------- | :--------------------- |
| SpatialTransformer     | STFT   | 0.0001 | 128                | 32         | 65.01                  |
| TemporalTransformer    | STFT   | 0.0007 | 64                 | 16         | 66.19                  |
| TemporalTransformer    | STFT   | 0.0007 | 128                | 32         | 65.90                  |
| SpatialCNNTransformer  | STFT   | 0.0007 | 128                | 32         | 70.59                  |
| SpatialCNNTransformer  | STFT   | 0.0001 | 128                | 32         | 70.20                  |
| TemporalCNNTransformer | STFT   | 0.0001 | 128                | 32         | 75.01                  |
| FusionCNNTransformer   | STFT   | 0.0007 | 128                | 32         | 71.97                  |
| FusionCNNTransformer   | STFT   | 0.0001 | 128                | 32         | 71.75                  |

**Variant 3**
| Model                  | Metoda | WU      | Liczba Komponentów | Wielkość Patchy | Średnia Dokładność (%) |
| :--------------------- | :----- | :------ | :----------------- | :-------------- | :--------------------- |
| SpatialTransformer     | CSP    | 0.00001 | 8                  | 16              | 68.09                  |
| SpatialTransformer     | CSP    | 0.001   | 8                  | 16              | 65.69                  |
| TemporalTransformer    | CSP    | 0.00001 | 6                  | 16              | 68.37                  |
| TemporalTransformer    | CSP    | 0.0007  | 6                  | 16              | 66.80                  |
| SpatialCNNTransformer  | CSP    | 0.0001  | 4                  | 32              | 69.27                  |
| TemporalCNNTransformer | CSP    | 0.0007  | 4                  | 32              | 69.37                  |
| TemporalCNNTransformer | CSP    | 0.001   | 4                  | 16              | 68.98                  |
| FusionCNNTransformer   | CSP    | 0.007   | 6                  | 16              | 69.55                  |
| FusionCNNTransformer   | CSP    | 0.0001  | 4                  | 32              | 62.88                  |

**Variant 4**
| Model                  | Metoda | Współczynnik Uczenia | Typ Falki | Średnia Dokładność (%) |
| :--------------------- | :----- | :------------------- | :-------- | :--------------------- |
| SpatialTransformer     | WPT    | 0.001                | db4       | 64.15                  |
| SpatialTransformer     | WPT    | 0.0001               | db4       | 62.40                  |
| TemporalTransformer    | WPT    | 0.0001               | coif3     | 74.44                  |
| TemporalTransformer    | WPT    | 0.0001               | coif3     | 73.70                  |
| SpatialCNNTransformer  | WPT    | 0.0001               | coif3     | 70.18                  |
| SpatialCNNTransformer  | WPT    | 0.001                | db4       | 68.30                  |
| TemporalCNNTransformer | WPT    | 0.0001               | coif3     | 74.92                  |
| FusionCNNTransformer   | WPT    | 0.0001               | db6       | 71.20                  |

**Variant 5**
| Model                  | Metoda | Współczynnik Uczenia | Średnia Dokładność (%) |
| :--------------------- | :----- | :------------------- | :--------------------- |
| SpatialCNNTransformer  | cnn    | 0.0007               | 72.05                  |
| SpatialCNNTransformer  | cnn    | 0.0001               | 70.47                  |
| TemporalCNNTransformer | cnn    | 0.0001               | 75.23                  |
| FusionCNNTransformer   | cnn    | 0.0001               | 71.48                  |

**Variant 6**
| Model                  | Metoda | Współczynnik Uczenia | Średnia Dokładność (%) |
| :--------------------- | :----- | :------------------- | :--------------------- |
| SpatialTransformer     | raw    | 0.0001               | 61.43                  |
| SpatialTransformer     | raw    | 0.0007               | 60.10                  |
| TemporalTransformer    | raw    | 0.0001               | 81.56                  |
| TemporalTransformer    | raw    | 0.0007               | 75.24                  |
| SpatialCNNTransformer  | raw    | 0.0007               | 77.19                  |
| TemporalCNNTransformer | raw    | 0.0001               | 79.39                  |
| FusionCNNTransformer   | raw    | 0.0001               | 78.48                  |

**Variant 7**
| Model                  | Metoda | WU     | Liczba punktów FFT | Skok Ramki | Średnia Dokładność (%) |
| :--------------------- | :----- | :----- | :----------------- | :--------- | :--------------------- |
| SpatialTransformer     | STFT   | 0.0007 | 128                | 32         | 67.27                  |
| TemporalTransformer    | STFT   | 0.0001 | 128                | 32         | 66.24                  |
| SpatialCNNTransformer  | STFT   | 0.0007 | 128                | 32         | 78.37                  |
| SpatialCNNTransformer  | STFT   | 0.0001 | 128                | 32         | 78.25                  |
| TemporalCNNTransformer | STFT   | 0.0001 | 64                 | 16         | 82.22                  |
| FusionCNNTransformer   | STFT   | 0.0001 | 128                | 32         | 79.71                  |
| FusionCNNTransformer   | STFT   | 0.0007 | 128                | 32         | 78.46                  |


**Variant 8**
| Model                  | Metoda | WU      | Liczba Komponentów | Wielkość Patchy | Średnia Dokładność (%) |
| :--------------------- | :----- | :------ | :----------------- | :-------------- | :--------------------- |
| SpatialTransformer     | CSP    | 0.00001 | 4                  | 16              | 69.41                  |
| SpatialTransformer     | CSP    | 0.00001 | 4                  | 32              | 68.19                  |
| SpatialTransformer     | CSP    | 0.001   | 6                  | 16              | 67.39                  |
| SpatialTransformer     | CSP    | 0.001   | 4                  | 32              | 65.60                  |
| TemporalTransformer    | CSP    | 0.00001 | 4                  | 32              | 69.18                  |
| TemporalTransformer    | CSP    | 0.0007  | 6                  | 16              | 66.26                  |
| SpatialCNNTransformer  | CSP    | 0.00001 | 4                  | 16              | 67.70                  |
| SpatialCNNTransformer  | CSP    | 0.0001  | 6                  | 16              | 66.89                  |
| TemporalCNNTransformer | CSP    | 0.00001 | 4                  | 16              | 67.80                  |
| FusionCNNTransformer   | CSP    | 0.0001  | 6                  | 16              | 65.78                  |
| FusionCNNTransformer   | CSP    | 0.007   | 6                  | 16              | 65.24                  |

**Variant 9**
| Model                  | Metoda | WU     | Typ Falki | Średnia Dokładność (%) |
| :--------------------- | :----- | :----- | :-------- | :--------------------- |
| SpatialTransformer     | WPT    | 0.001  | db4       | 64.78                  |
| SpatialTransformer     | WPT    | 0.0001 | db4       | 63.65                  |
| TemporalTransformer    | WPT    | 0.0001 | coif3     | 81.29                  |
| SpatialCNNTransformer  | WPT    | 0.0001 | coif3     | 74.26                  |
| TemporalCNNTransformer | WPT    | 0.0001 | coif3     | 81.68                  |
| FusionCNNTransformer   | WPT    | 0.0001 | db6       | 78.91                  |

**Variant 10**
| Model                  | Metoda | Współczynnik Uczenia | Średnia Dokładność (%) |
| :--------------------- | :----- | :------------------- | :--------------------- |
| SpatialCNNTransformer  | cnn    | 0.0007               | 79.93                  |
| SpatialCNNTransformer  | cnn    | 0.0001               | 78.96                  |
| TemporalCNNTransformer | cnn    | 0.0001               | 82.45                  |
| FusionCNNTransformer   | cnn    | 0.0001               | 79.02                  |

