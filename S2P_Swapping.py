import skrf as rf
import tkinter as tk
import os
from tkinter import filedialog, messagebox

# Global variable for the main Tkinter window instance
root = None

def get_unique_directory_path(base_dir, folder_name="Swapped"):
    """
    Generates a unique directory path by appending a number if the folder already exists.
    """
    counter = 0
    target_dir = os.path.join(base_dir, folder_name)

    # Check for existing folders and increment the counter
    while os.path.exists(target_dir):
        counter += 1
        target_dir = os.path.join(base_dir, f"{folder_name}_{counter}")
        
    return target_dir

def process_selected_files(file_paths):
    """
    Reads the list of selected s2p files, flips them, and saves them into a 
    new 'Swapped' folder within the source directory.
    """
    if not file_paths:
        messagebox.showinfo("Info", "No files selected for processing.")
        if root:
            root.destroy()
        return

    # Determine a common base directory for all selected files
    # (assuming all files are in the same or similar locations for this feature)
    # Use the directory of the first file as the reference base directory
    base_directory = os.path.dirname(file_paths[0])
    
    # Get a unique 'Swapped' directory path
    output_directory = get_unique_directory_path(base_directory)
    
    try:
        # Create the new directory
        os.makedirs(output_directory, exist_ok=True)
        print(f"Created output directory: {output_directory}")
    except OSError as e:
        messagebox.showerror("Error", f"Failed to create directory {output_directory}: {e}")
        if root:
            root.destroy()
        return

    processed_count = 0
    errors = []

    for file_path in file_paths:
        try:
            network = rf.Network(file_path)
            # Swapping only 2 port network, check for 2 port network
            if network.nports != 2:
                errors.append(f"Skipping {os.path.basename(file_path)}: not a 2-port network.")
                continue
            
            # Flip the network in place using the flip() method in Sckit-RF
            network.flip() 
            
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            swapped_name = f"{name}_Swapped{ext}"
            
            # Save the file into the new output directory
            new_file_path = os.path.join(output_directory, swapped_name)
            
            #network.write_touchstone(filename=new_file_path, write_z0=True) - REMOVED 1/16/25
            #removing write_z0 = true, to lower the swapped file size, considering the input file is already 50 ohm based. Also help to save in the format # Hz S  dB   R 50 which helps in loading in the VNA fixture removal.
            
            network.write_touchstone(filename=new_file_path)
            processed_count += 1
            print(f"Processed: {filename} -> Saved in: {os.path.basename(output_directory)}")

        except Exception as e:
            errors.append(f"Error processing {os.path.basename(file_path)}: {e}")

    # Call display results and pass the root window instance to close it
    display_results(processed_count, errors, root, output_directory)


def select_files():
    """
    Opens a dialog for the user to select multiple individual s2p files.
    """
    file_paths = filedialog.askopenfilenames(
        title="Select S2P Files to Flip",
        filetypes=(("S2P files", "*.s2p"), ("All files", "*.*"))
    )
    
    if file_paths:
        # Pass the selected paths to the processing function
        process_selected_files(file_paths)
    else:
        messagebox.showinfo("Info", "File selection cancelled.")
        # If cancelled, close the GUI as the user is done
        if root:
            root.destroy()


def display_results(count, errors, window_instance, output_dir=None):
    """
    Shows a summary dialog of the processing results, then closes the GUI window.
    Includes info about the output directory.
    """
    result_message = f"Successfully flipped and saved {count} file(s)."
    if output_dir:
        result_message += f"\nFiles saved in the new folder:\n{output_dir}"
        
    if errors:
        result_message += f"\n\nErrors occurred for the following files:\n" + "\n".join(errors)

    # Show the results messagebox (this is a blocking call)
    messagebox.showinfo("Processing Complete", result_message)
    
    # After the user clicks OK on the results message box, destroy the main Tkinter window
    if window_instance:
        window_instance.destroy()


# --- Tkinter User Interface Setup ---

# Initialize the main window and assign it to the global 'root' variable
root = tk.Tk()
root.title("S2P File Flipper Utility (Multi-File Select)")
root.geometry("450x120")

label = tk.Label(root, text="Select multiple individual S2P files using the button below:")
label.pack(pady=15)

# Button to trigger the file selection dialog
select_files_button = tk.Button(root, text="Select Files and Process", command=select_files)
select_files_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
