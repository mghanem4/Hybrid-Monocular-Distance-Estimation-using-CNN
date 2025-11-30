import numpy as np
import matplotlib.pyplot as plt
from dataset import KittiObjectDataset 


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#  THIS SOURCE FILE ONLY GETS THE IMAGE SIZE STATS FROM DATASET. NOTHING RELATED TO MODEL.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

def analyze_crop_sizes():
    print("Loading Dataset metadata...")
    # Initialize dataset
    dataset = KittiObjectDataset(mode='train')
    
    widths = []
    heights = []

    print(f"Analyzing {len(dataset.objects)} objects...")
    
    for obj in dataset.objects:
        # box format is [x1, y1, x2, y2]
        x1, y1, x2, y2 = obj['box']
        
        w = x2 - x1
        h = y2 - y1
        
        widths.append(w)
        heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    # Calculate Statistics
    print("-" * 30)
    print(f"Count:  {len(widths)}")
    print(f"WIDTH:  Min: {widths.min():.1f}, Max: {widths.max():.1f}, Avg: {np.mean(widths):.1f}, Median: {np.median(widths):.1f}")
    print(f"HEIGHT: Min: {heights.min():.1f}, Max: {heights.max():.1f}, Avg: {np.mean(heights):.1f}, Median: {np.median(heights):.1f}")
    print("-" * 30)

    # generally, we want to cover about 80-90% of cases without upscaling too much
    suggested_h = int(np.percentile(heights, 75)) # 75th percentile
    print(f"Data suggests a resolution around: {suggested_h} x {suggested_h}")

    # Plot histogram to visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Object Widths')
    
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Object Heights')
    # plt.show() #wont work on linux vm
    plt.savefig('distribution_plot.png')
    print("Success! Plot saved to 'distribution_plot.png' in your current folder.")

if __name__ == "__main__":
    analyze_crop_sizes()