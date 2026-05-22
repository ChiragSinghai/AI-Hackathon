import streamlit as st
import json
import pandas as pd
from datetime import datetime, timedelta
from rag_pipeline import process_form, ask_question
from utils import validate_field, get_suggestions, auto_fill_fields
import time
import os

pageTitle= "AI-Powered Student Visa Application"

# Page configuration
st.set_page_config(
    page_title=pageTitle,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .section-header {
        background-color: #e8f4f8;
        padding: 10px 15px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0 15px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .form-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Load form schema
@st.cache_resource
def load_form_schema():
    with open('form_schema.json', 'r') as f:
        return json.load(f)

schema = load_form_schema()

# Initialize session state
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

if 'errors' not in st.session_state:
    st.session_state.errors = {}

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {}

if 'auto_filled_fields' not in st.session_state:
    st.session_state.auto_filled_fields = set()

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if 'completion_metric' not in st.session_state:
    st.session_state.completion_metric = {"start_time": datetime.now(), "errors": 0}

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None

if 'pdf_qa_history' not in st.session_state:
    st.session_state.pdf_qa_history = []

# Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown(f"<div class='main-header'><h1>🎓 AI-Powered Student Visa Application</h1></div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
This form uses AI assistance to guide you through the application process. 
Get real-time suggestions, auto-fill capabilities, and instant validation.
</div>
""", unsafe_allow_html=True)

# Sidebar with metrics
with st.sidebar:
    st.header("📊 Form Metrics")
    
    # Calculate completion percentage
    total_fields = sum(len(section['fields']) for section in schema['sections'])
    filled_fields = len([k for k, v in st.session_state.form_data.items() if v and v != ""])
    completion_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
    
    st.metric("Form Completion", f"{completion_percentage:.0f}%", f"{filled_fields}/{total_fields} fields")
    st.progress(completion_percentage / 100)
    
    st.divider()
    
    # Errors metric
    error_count = len(st.session_state.errors)
    st.metric("Validation Errors", error_count, delta=None, delta_color="inverse")
    
    st.divider()
    
    # Time elapsed
    elapsed = datetime.now() - st.session_state.completion_metric['start_time']
    st.metric("Time Spent", f"{elapsed.seconds // 60} min {elapsed.seconds % 60} sec")
    
    st.divider()
    
    # AI Status
    api_key = os.getenv("OPENAI_API_KEY")
    genai_api_key = os.getenv("GENAI_LAB_API_KEY")
    
    if genai_api_key:
        st.success("✅ AI Suggestions: **ENABLED**", icon="🤖")
        st.caption("🎯 TCS GenAI Lab (DeepSeek-V3)")
    elif api_key:
        st.success("✅ AI Suggestions: **ENABLED**", icon="🤖")
        st.caption("🔄 OpenAI (Fallback)")
    else:
        st.warning("⚠️ AI Suggestions: **DISABLED**", icon="🤖")
        st.caption("Configure GENAI_LAB_API_KEY or OPENAI_API_KEY in .env")
    
    st.divider()
    
    # Help & Info
    st.subheader("ℹ️ Help")
    st.info("""
    **Tips for filling the form:**
    - Fields marked with * are required
    - AI suggestions appear next to eligible fields
    - Auto-fill is automatic for dependent fields
    - All fields are validated in real-time
    """)

# Main form area
tab1, tab2, tab3, tab4 = st.tabs(["📝 Application Form", "📤 Summary & Submit", "📊 Help & Guide", "📄 Upload & Ask PDF"])

with tab1:
    st.markdown("### Please Fill in All Required Fields")
    
    # Render form sections
    for section in schema['sections']:
        st.markdown(f"<div class='section-header'><h3>{section['section_name']}</h3></div>", unsafe_allow_html=True)
        
        section_cols = st.columns(1)
        
        for field in section['fields']:
            field_id = field['field_id']
            required_mark = " *" if field['required'] else ""
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Render based on field type
                if field['type'] == 'text':
                    st.session_state.form_data[field_id] = st.text_input(
                        f"{field['label']}{required_mark}",
                        value=st.session_state.form_data.get(field_id, ''),
                        placeholder=field.get('placeholder', ''),
                        help=field.get('help_text', ''),
                        key=f"input_{field_id}"
                    )
                
                elif field['type'] == 'email':
                    st.session_state.form_data[field_id] = st.text_input(
                        f"{field['label']}{required_mark}",
                        value=st.session_state.form_data.get(field_id, ''),
                        placeholder=field.get('placeholder', ''),
                        help=field.get('help_text', ''),
                        key=f"input_{field_id}"
                    )
                
                elif field['type'] == 'number':
                    try:
                        value = float(st.session_state.form_data.get(field_id, ''))
                    except (ValueError, TypeError):
                        value = None
                    
                    st.session_state.form_data[field_id] = st.number_input(
                        f"{field['label']}{required_mark}",
                        value=value,
                        placeholder=field.get('placeholder', ''),
                        help=field.get('help_text', ''),
                        key=f"input_{field_id}"
                    )
                
                elif field['type'] == 'date':
                    value = st.session_state.form_data.get(field_id, None)
                    
                    # Convert string to date if needed
                    if value and isinstance(value, str):
                        try:
                            value = datetime.strptime(value, '%Y-%m-%d').date()
                        except:
                            value = None
                    elif isinstance(value, datetime):
                        value = value.date()
                    elif not value:
                        value = None
                    
                    selected_date = st.date_input(
                        f"{field['label']}{required_mark}",
                        value=value,
                        help=field.get('help_text', ''),
                        key=f"input_{field_id}"
                    )
                    
                    # Store as string in YYYY-MM-DD format
                    if selected_date:
                        st.session_state.form_data[field_id] = selected_date.strftime('%Y-%m-%d')
                    else:
                        st.session_state.form_data[field_id] = None
                
                elif field['type'] == 'select':
                    current_value = st.session_state.form_data.get(field_id, '')
                    st.session_state.form_data[field_id] = st.selectbox(
                        f"{field['label']}{required_mark}",
                        options=[''] + field['options'],
                        index=[''] + field['options'].index(current_value) if current_value in field['options'] else 0,
                        help=field.get('help_text', ''),
                        key=f"input_{field_id}"
                    )
                
                elif field['type'] == 'textarea':
                    st.session_state.form_data[field_id] = st.text_area(
                        f"{field['label']}{required_mark}",
                        value=st.session_state.form_data.get(field_id, ''),
                        placeholder=field.get('placeholder', ''),
                        help=field.get('help_text', ''),
                        height=100,
                        key=f"input_{field_id}"
                    )
            
            # AI Suggestions column
            with col2:
                if st.button("💡 Suggest", key=f"suggest_{field_id}", help="Get AI suggestion for this field"):
                    with st.spinner("🤖 Getting AI suggestion..."):
                        suggestion = get_suggestions(field, pageTitle)
                        if suggestion:
                            st.session_state.suggestions[field_id] = suggestion
            
            # Display suggestions and errors
            if field_id in st.session_state.suggestions:
                st.markdown(f"""
                <div class='info-box'>
                💡 <b>Suggestion:</b> {st.session_state.suggestions[field_id]}
                </div>
                """, unsafe_allow_html=True)
            
            # Validation
            if field_id in st.session_state.form_data and st.session_state.form_data[field_id]:
                validation_result = validate_field(field, st.session_state.form_data[field_id])
                
                if not validation_result['valid']:
                    st.session_state.errors[field_id] = validation_result['message']
                    st.markdown(f"""
                    <div class='error-box'>
                    ❌ {validation_result['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if field_id in st.session_state.errors:
                        del st.session_state.errors[field_id]
                    st.markdown(f"""
                    <div class='success-box'>
                    ✅ Valid
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()

with tab2:
    st.markdown("### 📋 Form Summary")
    
    # Display filled data
    filled_data = {k: v for k, v in st.session_state.form_data.items() if v}
    
    if filled_data:
        st.json(filled_data)
    else:
        st.info("No data filled yet. Start filling the form to see summary.")
    
    st.divider()
    
    # Validation before submission
    st.markdown("### ✅ Validation Status")
    
    required_fields = []
    for section in schema['sections']:
        for field in section['fields']:
            if field['required']:
                required_fields.append(field['field_id'])
    
    missing_fields = [f for f in required_fields if not st.session_state.form_data.get(f)]
    errors_count = len(st.session_state.errors)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Required Fields", f"{len(required_fields) - len(missing_fields)}/{len(required_fields)}")
    
    with col2:
        st.metric("Validation Errors", errors_count)
    
    with col3:
        st.metric("Completion", f"{completion_percentage:.0f}%")
    
    if missing_fields:
        st.warning(f"⚠️ Missing {len(missing_fields)} required field(s): {', '.join(missing_fields)}")
    
    if errors_count > 0:
        st.error(f"❌ {errors_count} validation error(s) found. Please fix them before submitting.")
    
    st.divider()
    
    # Submit button
    if st.button("🚀 Submit Application", key="submit_btn", type="primary"):
        if not missing_fields and errors_count == 0:
            st.session_state.submitted = True
            st.session_state.completion_metric['end_time'] = datetime.now()
            
            st.markdown("""
            <div class='success-box'>
            <h3>✅ Application Submitted Successfully!</h3>
            <p>Your Student Visa Application has been submitted. You will receive a confirmation email shortly.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            st.markdown("### 📊 Submission Metrics")
            elapsed = st.session_state.completion_metric.get('end_time', datetime.now()) - st.session_state.completion_metric['start_time']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Time", f"{elapsed.seconds // 60} min")
            with col2:
                st.metric("Fields Filled", len(filled_data))
            with col3:
                st.metric("Errors During Process", st.session_state.completion_metric.get('errors', 0))
            with col4:
                st.metric("Final Score", "100%", delta="✅")
        else:
            st.error("Please fix all errors and fill required fields before submitting.")

with tab3:
    st.markdown("### ❓ Help & Guidance")
    
    st.markdown("""
    #### How to Use the AI Assistant
    
    **1. Field-Level Help 💡**
    - Click the "Suggest" button next to any field to get AI-powered suggestions
    - Suggestions are context-aware and based on your previous entries
    
    **2. Auto-Fill Capability 🚀**
    - Some fields automatically fill based on related fields
    - Example: If you enter your nationality, the country field auto-fills
    
    **3. Real-Time Validation ✅**
    - All fields are validated as you type
    - Error messages appear immediately if validation fails
    
    **4. Form Completion Tracking 📊**
    - Check your progress in the sidebar metrics
    - See how many fields are filled and how many errors remain
    
    #### Common Tips
    
    - **Date Format:** Always use YYYY-MM-DD format
    - **Phone Number:** Include country code (e.g., +91 for India)
    - **Passport:** Ensure passport validity is at least 6 months from visa issue date
    - **Financial Info:** Be accurate about income and savings
    
    #### What Information is Needed?
    
    | Section | Fields |
    |---------|--------|
    | Personal Information | Full name, DOB, Gender, Nationality, Email |
    | Contact Details | Phone, Address, City, Country |
    | Passport Info | Number, Issue/Expiry dates, Type |
    | Education | Qualification, University, Graduation date, Field of study |
    | Destination Program | University, Start date, Duration, Course type, Description |
    | Financial | Income, Savings, Funds provider |
    | Employment | Status, Job title |
    | Travel History | Visa rejections |
    | Additional | Health insurance |
    """)
    
    st.divider()
    
    st.markdown("""
    ### 🤖 About AI Assistance
    
    This application uses advanced AI (LangChain + GPT) to:
    - Understand form requirements
    - Generate contextual suggestions
    - Validate entries in real-time
    - Auto-fill related fields intelligently
    
    The AI learns from your inputs to provide better suggestions as you progress.
    """)


with tab4:
    st.markdown("### 📄 Upload Offline PDF Form & Ask Questions")
    
    st.markdown("""
    <div class='info-box'>
    Upload a government form PDF and ask AI questions about it. The form will be processed 
    and you can query specific sections, requirements, or get clarifications.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # PDF Upload Section
    st.markdown("#### 📥 Upload PDF Form")
    
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if uploaded_pdf:
        # Create uploads directory if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        # Save the uploaded file
        pdf_path = f"uploads/{uploaded_pdf.name}"
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_pdf.name}")
        st.session_state.pdf_filename = uploaded_pdf.name
        
        # Process the PDF
        if not st.session_state.pdf_processed:
            with st.spinner("🔄 Processing PDF form..."):
                try:
                    process_form(pdf_path)
                    st.session_state.pdf_processed = True
                    st.success("✅ PDF processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
        
        st.divider()
        
        # Q&A Section
        if st.session_state.pdf_processed:
            st.markdown("#### ❓ Ask Questions About the Form")
            
            query = st.text_input(
                "Enter your question about the form",
                placeholder="e.g., What documents are required? What is the deadline?"
            )
            
            if st.button("🔍 Get Answer", key="ask_pdf_btn", type="primary"):
                if query.strip():
                    with st.spinner("🤖 Finding answer..."):
                        try:
                            response = ask_question(query)
                            
                            # Store in history
                            st.session_state.pdf_qa_history.append({
                                "question": query,
                                "answer": response
                            })
                            
                            st.success("✅ Answer found!")
                            
                        except Exception as e:
                            st.error(f"❌ Error getting answer: {str(e)}")
                else:
                    st.warning("Please enter a question")
            
            st.divider()
            
            # Display Q&A History
            if st.session_state.pdf_qa_history:
                st.markdown("#### 📋 Q&A History")
                
                for idx, item in enumerate(st.session_state.pdf_qa_history, 1):
                    with st.expander(f"Q{idx}: {item['question'][:60]}..."):
                        st.markdown("**Question:**")
                        st.write(item['question'])
                        st.markdown("**Answer:**")
                        st.write(item['answer'])
                
                # Clear history button
                if st.button("🗑️ Clear History", key="clear_history"):
                    st.session_state.pdf_qa_history = []
                    st.rerun()
    
    else:
        st.info("👆 Upload a PDF form to get started")

# Footer
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("""
    <p style='text-align: center; color: gray;'>
    🎓 Student Visa Application | Powered by AI 🤖<br>
    <small>Reducing form completion time and errors</small>
    </p>
    """, unsafe_allow_html=True)
