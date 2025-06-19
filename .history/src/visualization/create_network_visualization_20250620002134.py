import matplotlib.pyplot as plt
import os

# 출력 디렉토리 생성
output_dir = "../../output/picture/presentation"
os.makedirs(output_dir, exist_ok=True)

def create_neural_network_visualization():
    """
    신경망 구조를 영문으로 시각화하여 저장합니다.
    """
    # Set up the figure
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
    
    # Define layers and node counts
    layers = [
        {'name': 'Input', 'nodes': 12, 'color': 'lightblue'},
        {'name': 'Hidden 1', 'nodes': 256, 'color': 'lightgreen'},
        {'name': 'Hidden 2', 'nodes': 128, 'color': 'lightgreen'},
        {'name': 'Hidden 3', 'nodes': 64, 'color': 'lightgreen'},
        {'name': 'Hidden 4', 'nodes': 32, 'color': 'lightgreen'},
        {'name': 'Output', 'nodes': 1, 'color': 'salmon'}
    ]
    
    # Calculate positions
    layer_width = 2.0  # Layer spacing
    x_positions = [i * layer_width * 2 for i in range(len(layers))]
    
    # Constants for display
    max_visible_nodes = 10  # Max nodes to display per layer
    display_height = 12  # Fixed display height
    node_radius = 0.3  # Node size
    
    # Draw each layer
    for i, (x, layer) in enumerate(zip(x_positions, layers)):
        # Layer name
        ax.text(x, -2.0, layer['name'], ha='center', fontsize=13, weight='bold', fontname='Arial')
        
        # Layer background box
        box_width = 1.2
        box_height = display_height + 1
        rect = plt.Rectangle(
            (x-box_width/2, -1), box_width, box_height, 
            fill=True, alpha=0.1, color=layer['color'], 
            edgecolor='gray', linestyle='--'
        )
        ax.add_patch(rect)
        
        # Node count text
        ax.text(
            x, display_height + 1, f"Nodes: {layer['nodes']}", 
            ha='center', fontsize=11, fontname='Arial',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
        )
        
        # Draw nodes
        nodes_to_draw = min(layer['nodes'], max_visible_nodes)
        
        # Calculate spacing between nodes
        if nodes_to_draw <= 1:
            spacing = 0
        else:
            spacing = display_height / (nodes_to_draw + 1)
        
        for j in range(nodes_to_draw):
            # Node position
            y = spacing * (j + 1)
            
            if nodes_to_draw < layer['nodes'] and j == nodes_to_draw - 1:
                # Ellipsis for omitted nodes
                ax.text(x, y, '⋮', ha='center', va='center', fontsize=24, weight='bold')
            else:
                # Draw node
                circle = plt.Circle(
                    (x, y), node_radius, 
                    facecolor=layer['color'], edgecolor='blue', alpha=0.8, zorder=10
                )
                ax.add_patch(circle)
            
            # Connect to previous layer
            if i > 0:
                prev_layer = layers[i-1]
                prev_x = x_positions[i-1]
                prev_nodes_to_draw = min(prev_layer['nodes'], max_visible_nodes)
                
                if prev_nodes_to_draw <= 1:
                    prev_spacing = 0
                else:
                    prev_spacing = display_height / (prev_nodes_to_draw + 1)
                
                for k in range(prev_nodes_to_draw):
                    prev_y = prev_spacing * (k + 1)
                    
                    if prev_nodes_to_draw < prev_layer['nodes'] and k == prev_nodes_to_draw - 1:
                        continue  # Skip connections to ellipsis
                    
                    # Draw connections (subset for clarity)
                    if (j % 3 == 0 and k % 3 == 0) or (i == len(layers) - 1) or (nodes_to_draw <= 3):
                        ax.plot(
                            [prev_x, x], [prev_y, y], 
                            'gray', alpha=0.15, linewidth=0.5, zorder=1
                        )
        
        # Add activation functions and other components
        if 0 < i < len(layers) - 1:
            components = [
                {'text': 'ReLU', 'color': 'green', 'y_offset': 0},
                {'text': f'Dropout ({0.3 if i==1 else 0.2 if i==2 else 0.1})', 'color': 'red', 'y_offset': -1.2},
                {'text': 'BatchNorm', 'color': 'purple', 'y_offset': -2.4}
            ]
            
            for component in components:
                # Position between layers
                mid_x = (x_positions[i-1] + x) / 2
                ax.text(
                    mid_x, display_height/2 + component['y_offset'], component['text'], 
                    ha='center', color=component['color'], fontsize=9, fontname='Arial',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                    rotation=0, zorder=20
                )
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Input Layer'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Hidden Layers'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.05))
    
    # Additional explanation
    plt.figtext(
        0.5, 0.02, 
        '* A simplified diagram of the implemented DNN model architecture',
        ha='center', fontsize=10, style='italic', fontname='Arial'
    )
    
    # Title and layout
    plt.title('Deep Neural Network (DNN) Model Architecture', fontsize=18, pad=20, fontname='Arial')
    ax.set_xlim(-1.0, max(x_positions) + 1.0)
    ax.set_ylim(-3.0, display_height + 3)
    ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_path = os.path.join(output_dir, 'neural_network_architecture_fixed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Neural network architecture visualization saved to: {output_path}")

if __name__ == "__main__":
    create_neural_network_visualization()
