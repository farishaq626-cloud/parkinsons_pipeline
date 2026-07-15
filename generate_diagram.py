import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, xy, text, color):
    # Draw a rounded box
    rect = patches.FancyBboxPatch(xy, 0.2, 0.12, boxstyle="round,pad=0.05", 
                                  linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    # Add text
    ax.text(xy[0]+0.1, xy[1]+0.06, text, ha='center', va='center', fontsize=9, fontweight='bold')

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Coordinates (x, y)
    box_positions = [
        (0.05, 0.4, "Data Ingestion\n(UCI Dataset)", "#E1F5FE"),
        (0.3, 0.4, "Processing Engine\n(Cleaning/Scaling)", "#E8F5E9"),
        (0.55, 0.4, "Model Layer\n(Random Forest)", "#FFF3E0"),
        (0.8, 0.4, "Output\n(Metrics/Plots)", "#F5F5F5")
    ]
    
    # Draw boxes
    for pos in box_positions:
        draw_box(ax, (pos[0], pos[1]), pos[2], pos[3])
        
    # Draw arrows
    for i in range(len(box_positions) - 1):
        ax.annotate("", xy=(box_positions[i+1][0], 0.46), xytext=(box_positions[i][0]+0.2, 0.46),
                    arrowprops=dict(arrowstyle="->", lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("Parkinson's Prediction Pipeline Architecture")
    plt.tight_layout()
    plt.savefig('pipeline_architecture.png', dpi=300)
    print("Successfully generated: pipeline_architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()