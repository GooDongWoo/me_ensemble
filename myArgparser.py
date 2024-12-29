# Add these imports at the top of the file, after the existing imports
import argparse
import ast

def parse_list_arg(arg_str):
    """
    Parse string argument into a list of integers.
    Handles both quoted and unquoted list formats.
    """
    try:
        # Remove any outer quotes if present
        arg_str = arg_str.strip('"\'')
        
        # If input is formatted like a list [1,2,3]
        if arg_str.startswith('[') and arg_str.endswith(']'):
            # Parse string as a literal Python expression
            result = ast.literal_eval(arg_str)
        else:
            # If input is just numbers separated by commas: 1,2,3
            result = [int(x.strip()) for x in arg_str.split(',')]
        
        # Validate all elements are integers
        if not all(isinstance(x, int) for x in result):
            raise ValueError("All elements must be integers")
            
        # Validate all elements are in range [0-9]
        if not all(0 <= x <= 9 for x in result):
            raise ValueError("All elements must be between 0 and 9")
            
        return result
        
    except Exception as e:
        print(f"Error parsing argument: {e}")
        print("process as no argument")
        return None

def parse_args(mydefault):
    """
    Parse command line arguments for the training script.
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Training script for multi-exit neural networks')
    
    # Add argument for unfreeze_ees_list
    parser.add_argument('-u', '--unfreeze-exits',
        type=str,
        default=None,
        help='List of exit layers to unfreeze. Can be specified in multiple formats:\n'\
            '1. With brackets: [4] or [0,1,2]\n'\
            '2. Without brackets: 4 or 0,1,2\n'\
            f'3. Default is {mydefault}\n'\
            '4. Note: Exits are indexed from 0 to 9\n'\
            '5. Don\'t space between commas\n'
    )
    
    args = parser.parse_args()
    
    # Parse the unfreeze-exits argument if provided
    if args.unfreeze_exits is not None:
        args.unfreeze_exits = parse_list_arg(args.unfreeze_exits)
    
    return args

# Modify the configuration parameters section
# Replace the existing unfreeze_ees_list definition with:
if __name__ == '__main__':
    my_default = [0,1,2,3,4,5,6,7,8,9]
    args = parse_args(my_default)
    unfreeze_ees_list = args.unfreeze_exits if args.unfreeze_exits is not None else [0,1,2,3,4,5,6,7,8,9]