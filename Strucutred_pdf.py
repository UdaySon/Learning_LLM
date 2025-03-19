import pdfplumber
import os

# Define the output directory for images
output_dir = r"C:\MachineLearning\pdf_AI\extracted_images"
os.makedirs(output_dir, exist_ok=True)

pdf_path = r'C:\MachineLearning\pdf_AI\AUTOSAR_SWS_CryptoDriver-R22-11.pdf'  # Change this to your PDF file path

structured_output = []

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        structured_output.append(f"\n=== Page {page_num} ===\n")

        # Extract text
        text = page.extract_text()
        if text:
            structured_output.append("\n**Text:**\n" + text)

        # Extract tables
        tables = page.extract_table()
        if tables:
            structured_output.append("\n**Table:**\n")
            for row in tables:
                for i in range(0,len(row)):
                    if(row[i] == None):
                        row[i] = ""
                cleaned_row = [' '.join(cell.splitlines()) if cell else '' for cell in row]
                structured_output.append(" | ".join(cleaned_row))

            # Generate underline for column headers
            underline = ["-" * len(cell) for cell in tables[0]]
            structured_output.insert(-len(tables), " | ".join(underline))  # Insert after headers

        # Extract images
        images = page.images
        if images:
            structured_output.append("\n**Images:**\n")
            for img_index, img in enumerate(images):
                img_obj = page.to_image()
                img_path = os.path.join(output_dir, f"page_{page_num}_img_{img_index}.png")
                img_obj.save(img_path, format="PNG")
                structured_output.append(f"[Image saved: {img_path}]")

# Save structured output to a file
output_txt = "\n".join(structured_output)
with open(r"C:\MachineLearning\pdf_AI\output.txt", "w", encoding="utf-8") as f:
    f.write(output_txt)

print("Extraction complete! Check 'output.txt' for text and tables, and 'extracted_images' for images.")
