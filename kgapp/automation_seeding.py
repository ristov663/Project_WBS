import os
import subprocess
import sys


# Function to execute a Python script
def run_script(script_path):
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"Successfully executed script: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")
        sys.exit(1)


# Main function to orchestrate the automation process
def main():
    # Define paths to the scripts
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_scraping_path = os.path.join(base_path, 'data_scraping.py')
    ontology_loader_path = os.path.join(base_path, 'ontology_loader.py')
    rdf_to_models_path = os.path.join(base_path, 'rdf_to_models.py')

    # 1. Execute data scraping script
    print("Starting data scraping...")
    run_script(data_scraping_path)

    # 2. Execute ontology loader script
    print("Starting ontology loading...")
    run_script(ontology_loader_path)

    # 3. Execute RDF to models conversion script
    print("Starting RDF to models conversion...")
    run_script(rdf_to_models_path)

    print("Automation completed successfully!")


# Entry point of the script
if __name__ == "__main__":
    main()
