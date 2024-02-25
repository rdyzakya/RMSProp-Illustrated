import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from loss import *
import cv2

def visualize_regression(df, m_sgd, m_rprop, m_rmsprop, c_sgd, c_rprop, c_rmsprop, x_range=(0, 1), num_points=100):
    # Generate x values
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate y values using the line equation y = mx + c
    y_values_sgd = m_sgd * x_values + c_sgd
    y_values_rprop = m_rprop * x_values + c_rprop
    y_values_rmsprop = m_rmsprop * x_values + c_rmsprop

    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot the line
    ax.scatter(df.iloc[:,0], df.iloc[:,1])
    ax.plot(x_values, y_values_sgd, label=f'SGD', color="red")
    ax.plot(x_values, y_values_rprop, label=f'Rprop', color="blue")
    ax.plot(x_values, y_values_rmsprop, label=f'RMSprop', color="green")
    
    # Add labels and legend
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Regression Visualization')
    ax.legend()
    
    # Show the plot
    ax.grid(True)
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    
    return fig

def visualize_classification(df, m_sgd, m_rprop, m_rmsprop, c_sgd, c_rprop, c_rmsprop, x_range=(0, 1), num_points=100):
    # Generate x values
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate y values using the sigmoid function
    y_values_sgd = 1 / (1 + np.exp(-(m_sgd * x_values + c_sgd)))
    y_values_rprop = 1 / (1 + np.exp(-(m_rprop * x_values + c_rprop)))
    y_values_rmsprop = 1 / (1 + np.exp(-(m_rmsprop * x_values + c_rmsprop)))

    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot the sigmoid function
    ax.scatter(df.iloc[:,0], df.iloc[:,1])
    ax.plot(x_values, y_values_sgd, label=f'SGD', color="red")
    ax.plot(x_values, y_values_rprop, label=f'Rprop', color="blue")
    ax.plot(x_values, y_values_rmsprop, label=f'RMSprop', color="green")
    
    # Add labels and legend
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Classification Visualization')
    ax.legend()
    
    # Show the plot
    ax.grid(True)
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)

    return fig

def visualize_loss(func, x_range=(-1,1), y_range=(-1,1), num_points=100):
    # Generate x and y values
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    
    # Create a meshgrid from x and y values
    X, Y = np.meshgrid(x_values, y_values)
    
    # Calculate the function values for each point in the meshgrid
    Z = func(X, Y)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('Slope') # X
    ax.set_ylabel('Intercept') # Y
    ax.set_zlabel('Loss') # Z
    ax.set_title('Optimization Visualization')
    
    # Show the plot
    return fig, ax

############################

def combine_images(image1, image2):
    # Get the size of the first image
    width1, height1 = image1.size

    # Get the size of the second image
    width2, height2 = image2.size

    # Calculate the size of the combined image
    new_width = max(width1, width2)
    new_height = height1 + height2

    # Create a new image with the calculated size
    combined_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste the first image on top
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the bottom
    combined_image.paste(image2, (0, height1))

    return combined_image

############################

def animate_gradient(data_index, dataloader, args, slope_range, intercept_range, history_sgd, history_rprop, history_rmsprop):
    frames = []

    loss_func = bce_loss if args.cls else mse_loss
    
    c_history_sgd = {
        "loss" : [],
        "slope" : [],
        "intercept" : []
    }
    
    c_history_rprop = {
        "loss" : [],
        "slope" : [],
        "intercept" : []
    }
    
    c_history_rmsprop = {
        "loss" : [],
        "slope" : [],
        "intercept" : []
    }
    
    df = dataloader.dataset.df
    x = df.iloc[data_index, 0]
    y = df.iloc[data_index, 1]
    
    for i in tqdm(range(args.epoch)):
        
        fig, ax = visualize_loss(lambda m, c: loss_func(x=x, y=y, m=m, c=c), x_range=slope_range, y_range=intercept_range)
        
        c_history_sgd["loss"].append(history_sgd["loss"][i][data_index])
        c_history_sgd["slope"].append(history_sgd["slope"][i][data_index])
        c_history_sgd["intercept"].append(history_sgd["intercept"][i][data_index])
        
        c_history_rprop["loss"].append(history_rprop["loss"][i][data_index])
        c_history_rprop["slope"].append(history_rprop["slope"][i][data_index])
        c_history_rprop["intercept"].append(history_rprop["intercept"][i][data_index])
        
        c_history_rmsprop["loss"].append(history_rmsprop["loss"][i][data_index])
        c_history_rmsprop["slope"].append(history_rmsprop["slope"][i][data_index])
        c_history_rmsprop["intercept"].append(history_rmsprop["intercept"][i][data_index])
        
        # trace line
        if i > 0:
            ax.plot(c_history_sgd["slope"], c_history_sgd["intercept"], c_history_sgd["loss"], alpha=0.5, color="red")
            ax.plot(c_history_rprop["slope"], c_history_rprop["intercept"], c_history_rprop["loss"], alpha=0.5, color="blue")
            ax.plot(c_history_rmsprop["slope"], c_history_rmsprop["intercept"], c_history_rmsprop["loss"], alpha=0.5, color="green")
        # dot
        ax.scatter(c_history_sgd["slope"][-1], c_history_sgd["intercept"][-1], c_history_sgd["loss"][-1], color="red", edgecolor="black", s=50, label="SGD")
        ax.scatter(c_history_rprop["slope"][-1], c_history_rprop["intercept"][-1], c_history_rprop["loss"][-1], color="blue", edgecolor="black", s=50, label="Rprop")
        ax.scatter(c_history_rmsprop["slope"][-1], c_history_rmsprop["intercept"][-1], c_history_rmsprop["loss"][-1], color="green", edgecolor="black", s=50, label="RMSprop")
        
        ax.legend()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        
        frames.append(pil_image)
    
    return frames

def animate_fit(data_index, dataloader, args, history_sgd, history_rprop, history_rmsprop):
    frames = []
    df = dataloader.dataset.df
    visualize_func = visualize_classification if args.cls else visualize_regression
    for i in tqdm(range(args.epoch)):
        m_sgd = history_sgd["slope"][i][data_index]
        m_rprop = history_rprop["slope"][i][data_index]
        m_rmsprop = history_rmsprop["slope"][i][data_index]

        c_sgd = history_sgd["intercept"][i][data_index]
        c_rprop = history_rprop["intercept"][i][data_index]
        c_rmsprop = history_rmsprop["intercept"][i][data_index]

        fig = visualize_func(df, m_sgd, m_rprop, m_rmsprop, c_sgd, c_rprop, c_rmsprop, x_range=(0, 1), num_points=100)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        frame = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

        frames.append(frame)
    
    return frames

############################

def create_video_from_images(image_list, output_path, fps):

    # Convert PIL images to NumPy arrays (OpenCV format)
    video_frames = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in image_list]

    # Get the height and width of the images
    height, width, layers = video_frames[0].shape

    # Create VideoWriter object
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Write each frame to the video
    for frame in video_frames:
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()