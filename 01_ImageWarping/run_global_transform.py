import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
   #transformed_image = cv2.resize(transformed_image, (int(transformed_image.shape[1]*scale), int(transformed_image.shape[0]*scale)))
    transformed_image = cv2.resize(transformed_image, dsize=None, fx=scale, fy=scale)
    #transformed_image[:] = int(scale*100)
    # Get the center of the image
    center = (transformed_image.shape[1] // 2, transformed_image.shape[0] // 2)

    #  # Create translation matrices
    # M_translate_to_center = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])
    # M_translate_back = np.float32([[1, 0, center[0]], [0, 1, center[1]]])
    
    # # Apply translation to center
    # transformed_image = cv2.warpAffine(transformed_image, M_translate_to_center, (transformed_image.shape[1], transformed_image.shape[0]))
    
    # # Apply scaling
    # transformed_image = cv2.resize(transformed_image, dsize=None, fx=scale, fy=scale)
    
    # # Apply translation back to original position
    # transformed_image = cv2.warpAffine(transformed_image, M_translate_back, (transformed_image.shape[1], transformed_image.shape[0]))
    
    
    # Apply rotation
    M = cv2.getRotationMatrix2D(center, rotation, 1)
    transformed_image = cv2.warpAffine(transformed_image, M, (transformed_image.shape[1], transformed_image.shape[0]))
    
    # Apply translation
    M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    transformed_image = cv2.warpAffine(transformed_image, M, (transformed_image.shape[1], transformed_image.shape[0]))

    # Apply horizontal flip
    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)


    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
