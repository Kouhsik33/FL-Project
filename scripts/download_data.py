"""Download and verify the public datasets.  Usage: python scripts/download_data.py [wdbc] [heart_disease]"""

from fedmed.data.download import main

if __name__ == "__main__":
    main()
