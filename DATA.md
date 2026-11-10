# Dataset

This project uses the Kaggle dataset **Astana Real Estate Dataset** by `turarr`.

Dataset link: https://www.kaggle.com/datasets/turarr/astana-real-estate-dataset

The notebook searches in this order: Kaggle's attached `/kaggle/input` files, an optional local `astana_real_estate_dataset.csv` in the project root, then an automatic download from `turarr/astana-real-estate-dataset` through KaggleHub. A local CSV is no longer required. The first download outside Kaggle requires internet access; public datasets generally do not require Kaggle credentials unless access is restricted by the owner.

The supplied optional local copy has 5,003 rows, 23 columns, and a non-missing `price` target. SHA-256: `3faebfe80cb2ede05007806b672cfcac62be86d0711fd8b95982b07d131c5c23`. It is ignored by Git.

This is a third-party collection of Krisha.kz advertisements, not an official government dataset. Check the source licence and redistribution terms before publishing or sharing a local CSV copy.
