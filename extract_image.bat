cd "C:\Users\yueny\OneDrive\Documents\Netvlad_3001\pytorch-NetVlad"

# Delete old folder if it exists
rmdir /S /Q zoo5 2>nul

# Create new structure
mkdir zoo5\train zoo5\val

# Copy & rename all images at once
robocopy places365_val\glacier       zoo5\train\arctic   *.jpg /E /MT /NFL /NDL /NJH /NJS
robocopy places365_val\forest_path   zoo5\train\forest   *.jpg /E /MT /NFL /NDL /NJH /NJS
robocopy places365_val\bamboo_forest zoo5\train\bamboo   *.jpg /E /MT /NFL /NDL /NJH /NJS
robocopy places365_val\field-wild     zoo5\train\savanna  *.jpg /E /MT /NFL /NDL /NJH /NJS
robocopy places365_val\desert-sand   zoo5\train\desert   *.jpg /E /MT /NFL /NDL /NJH /NJS