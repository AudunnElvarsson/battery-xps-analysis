"""
This is to test the xps_analysis package.
"""

import matplotlib.pyplot as plt
import xps_analysis.xps_utilities as xu

def main():
    """
    Main function to control what runs.
    """
    fig = plt.subplots()
    _ = fig
    xu.test()


if __name__ == "__main__":
    main()
