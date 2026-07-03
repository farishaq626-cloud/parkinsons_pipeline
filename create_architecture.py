import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, text):
    # This draws a professional-looking box for your architecture
    rect = patches.FancyBboxPatch((x, y), 2.5, 1, boxstyle="round,pad=0.2", 
                                  linewidth=1, edgecolor='black', facecolor='#e1f5fe')
    ax.add_patch(rect)
    ax.text(x + 1.25, y + 0.5, text, ha='center', va='center', fontsize=10)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 15)
ax.set_ylim(0, 2)
ax.axis('off')

# Draw the 4 main stages
draw_box(ax, 0, 0.5, "Data Sources\n(PPMI/Kaggle)")
draw_box(ax, 3.5, 0.5, "ETL Pipeline\n(Pandas)")
draw_box(ax, 7, 0.5, "Random Forest\nClassifier")
draw_box(ax, 10.5, 0.5, "FastAPI & \nStreamlit")

# Draw the connecting arrows
plt.annotate('', xy=(3.5, 1), xytext=(2.5, 1), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(7, 1), xytext=(6, 1), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(10.5, 1), xytext=(9.5, 1), arrowprops=dict(arrowstyle='->'))

# Save as PDF for your paper
plt.savefig('pipeline_architecture.pdf')
print("Figure 1 generated: pipeline_architecture.pdf")