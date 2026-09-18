
# Helper function to access table metadata quickly
def get_table_metadata(table_name, tables_metadata):
    for table in tables_metadata:
        if table['name'].lower() == table_name.lower():
            return table
    return None

# Extract relevant metadata
def extract_metadata(sql_cohort_preparation_json, json_metadata):
    extracted_metadata = {}

    # foreach relevant table in the SQL preparation JSON
    # eg: "relevant_tables": [{ "table_name": "supplemental_stroke_data",
    for table_info in sql_cohort_preparation_json['relevant_tables']:
        table_name = table_info['table_name']
        columns_needed = table_info['columns_needed']

        # Find table metadata for the given table name
        table_metadata = get_table_metadata(table_name, json_metadata['tables'])
        if not table_metadata:
            print(f"Table '{table_name}' not found in metadata.")
            continue

        extracted_metadata[table_name] = {}

        for col_name in columns_needed:
            # Find column metadata
            column_metadata = next((col for col in table_metadata['columns'] if col['name'].lower() == col_name.lower()), None)
            if not column_metadata:
                print(f"Column '{col_name}' not found in metadata for table '{table_name}'.")
                continue

            # Store metadata
            extracted_metadata[table_name][col_name] = {
                'null_count': column_metadata['null_count'],
                'empty_count': column_metadata['empty_count'],
                'distinct_count': column_metadata['distinct_count'],
                'top_values': column_metadata['top_values']
            }

    return extracted_metadata