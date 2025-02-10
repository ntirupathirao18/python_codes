import torch

def calculate_metrics(pred, gt, value):
    # True Positives (TP): pred value and gt value are both equal to the specified value
    TP = torch.sum(torch.logical_and(pred == value, gt == value))

    # False Positives (FP): pred value is equal to the specified value while gt value is not
    FP = torch.sum(torch.logical_and(pred == value, gt != value))

    # False Negatives (FN): pred value is not equal to the specified value while gt value is
    FN = torch.sum(torch.logical_and(pred != value, gt == value))

    return TP.item(), FP.item(), FN.item()

# Assuming pred and gt are PyTorch tensors of shape (1080, 1920)
# Replace pred_tensor and gt_tensor with your actual data
pred_tensor = torch.randint(0, 11, size=(1080, 1920))  # Example random prediction tensor
gt_tensor = torch.randint(0, 11, size=(1080, 1920))    # Example random ground truth tensor

# Generate values from 0 to 10 as a PyTorch tensor
values = torch.arange(11)

# Expand dimensions of pred_tensor and gt_tensor to perform element-wise comparison with values
pred_expanded = pred_tensor.unsqueeze(-1)
gt_expanded = gt_tensor.unsqueeze(-1)

# Compute TP, FP, FN for each value using tensor operations
TP = torch.sum(torch.logical_and(pred_expanded == values, gt_expanded == values), dim=(0, 1))
FP = torch.sum(torch.logical_and(pred_expanded == values, gt_expanded != values), dim=(0, 1))
FN = torch.sum(torch.logical_and(pred_expanded != values, gt_expanded == values), dim=(0, 1))

print("True Positives (TP):", TP.tolist())
print("False Positives (FP):", FP.tolist())
print("False Negatives (FN):", FN.tolist())



from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

def create_color_analysis_prompt(image_path):
    """
    Create structured prompts for color analysis using the color thesaurus reference
    """
    base_prompt = """Analyze the colors in this image using these specific color categories:

    1. Yellow family: ranging from lemon to mellow yellow
    2. Orange family: from bright orange to rust
    3. Red family: from scarlet to maroon
    4. Pink family: from hot pink to pale rose
    5. Violet family: from purple to lavender
    6. Blue family: from royal blue to sky blue
    7. Green family: from emerald to sage
    8. Brown family: from chocolate to tan
    9. Gray family: from charcoal to silver

    For each color you identify:
    1. Specify which color family it belongs to
    2. Describe its specific shade using the most accurate term
    3. Note its intensity (deep, medium, light)
    4. Mention if it's a pure shade or mixed with other tones
    """
    
    return base_prompt

def analyze_specific_shade(image_path, color_family):
    """
    Analyze a specific color family's shades in the image
    """
    color_families = {
        "yellow": ["lemon", "sunshine", "canary", "butter", "cream", "ivory", "mellow"],
        "orange": ["tangerine", "coral", "peach", "rust", "bronze", "amber"],
        "red": ["scarlet", "crimson", "ruby", "cherry", "wine", "maroon"],
        "pink": ["hot pink", "rose", "salmon", "blush", "flesh", "pale rose"],
        "violet": ["purple", "plum", "mauve", "lilac", "lavender"],
        "blue": ["royal blue", "navy", "azure", "cerulean", "sky blue", "powder blue"],
        "green": ["emerald", "forest", "olive", "sage", "mint", "seafoam"],
        "brown": ["chocolate", "coffee", "mocha", "copper", "caramel", "tan"],
        "gray": ["charcoal", "slate", "pewter", "stone", "silver", "ash"]
    }
    
    specific_prompt = f"""Focus on the {color_family} tones in this image.
    Reference shades for {color_family}: {', '.join(color_families[color_family])}
    
    Please identify:
    1. Which specific shade of {color_family} is present
    2. How it compares to the reference shades listed
    3. Whether it's a pure shade or mixed with other tones
    4. Its position in the {color_family} gradient (lighter/darker)"""
    
    return specific_prompt

def process_image_colors(image_path, model, tokenizer):
    """
    Process an image with Qwen 2 VL using the color thesaurus reference
    """
    image = Image.open(image_path)
    
    # General color analysis
    general_prompt = create_color_analysis_prompt(image_path)
    general_result = model.chat(tokenizer, general_prompt, history=[], images=image)
    
    # Store results
    color_analysis = {
        "general_analysis": general_result,
        "specific_analyses": {}
    }
    
    # Example of analyzing specific color families
    for color_family in ["blue", "green", "red"]:  # Add more as needed
        specific_prompt = analyze_specific_shade(image_path, color_family)
        specific_result = model.chat(tokenizer, specific_prompt, history=[], images=image)
        color_analysis["specific_analyses"][color_family] = specific_result
    
    return color_analysis

# Example usage
def example_color_analysis():
    # Initialize model (assuming already set up)
    model_name = "Qwen/Qwen2-VL"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    
    # Analyze image
    image_path = "your_image.jpg"
    results = process_image_colors(image_path, model, tokenizer)
    
    # Print results
    print("General Color Analysis:", results["general_analysis"])
    for color, analysis in results["specific_analyses"].items():
        print(f"\n{color.capitalize()} Analysis:", analysis)