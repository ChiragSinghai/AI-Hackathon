from rag_pipeline import ask_question2
import re

def validate_field(field, value):
    """Validates field based on type and constraints"""
    if not value and field['required']:
        return {'valid': False, 'message': f"{field['label']} is required"}
    
    if not value:
        return {'valid': True, 'message': 'Valid'}
    
    # Email validation
    if field['type'] == 'email':
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, str(value)):
            return {'valid': False, 'message': 'Invalid email format'}
    
    # Date validation
    if field['type'] == 'date':
        try:
            from datetime import datetime
            datetime.strptime(str(value), '%Y-%m-%d')
        except ValueError:
            return {'valid': False, 'message': 'Invalid date format (use YYYY-MM-DD)'}
    
    # Number validation
    if field['type'] == 'number':
        try:
            float(value)
        except ValueError:
            return {'valid': False, 'message': 'Invalid number'}
    
    return {'valid': True, 'message': 'Valid'}


def get_suggestions(field, context):
    """
    Gets AI suggestions for a field using the SLM
    context: pageTitle (e.g., "AI-Powered Student Visa Application")
    """
    # Create a specific query for this field
    query = f"""
    I'm filling out a form for: {context}
    
    Field: {field['label']}
    Field Type: {field['type']}
    Description: {field.get('help_text', 'N/A')}
    Required: {field['required']}
    
    Please provide a brief, helpful suggestion for what information should be entered in this field. 
    Be specific and practical. If there are any format requirements or important notes, mention them.
    Keep the response to 1-2 sentences maximum.
    """
    
    try:
        # Call the SLM through RAG pipeline
        suggestion = ask_question2(query)
        return suggestion
    except Exception as e:
        return f"Unable to generate suggestion: {str(e)}"


def auto_fill_fields(form_data, field_id, value):
    """Auto-fills related fields based on dependencies"""
    # Example: Auto-fill country based on nationality
    field_dependencies = {
        'nationality': 'country',  # If user selects nationality, suggest country
        'university_name': 'country_of_study'  # If university is selected, suggest country
    }
    
    auto_filled = {}
    
    if field_id in field_dependencies:
        dependent_field = field_dependencies[field_id]
        # You can add logic to auto-fill based on the value
        # This is where you'd make intelligent mappings
    
    return auto_filled