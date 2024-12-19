import streamlit as st
import os
import pandas as pd
import io
import csv
import pytesseract
from PIL import Image
import yt_dlp
import json
from datetime import datetime, timedelta
# Existing imports
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import urllib.parse
from urllib.parse import urlparse


class WebsiteProcessor:
    @staticmethod
    def is_valid_url(url):
        """Check if the URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    @staticmethod
    def extract_website_content(url):
        """
        Extract content from a website URL
        
        Args:
            url (str): Website URL
            
        Returns:
            tuple: Raw text and structured content dictionary
        """
        try:
            # Send request with headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()
            
            # Extract structured content
            content = {
                'title': soup.title.string if soup.title else '',
                'headings': [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
                'paragraphs': [p.text.strip() for p in soup.find_all('p') if p.text.strip()],
                'links': [{'text': a.text.strip(), 'href': a.get('href')} 
                         for a in soup.find_all('a') if a.text.strip() and a.get('href')],
                'images': [{'alt': img.get('alt', ''), 'src': img.get('src')} 
                          for img in soup.find_all('img') if img.get('src')]
            }
            
            # Combine paragraphs for raw text
            raw_text = '\n'.join(content['paragraphs'])
            
            return raw_text, content
            
        except Exception as e:
            st.error(f"Error extracting website content: {e}")
            return "", {}

    @staticmethod
    def analyze_website_structure(content):
        """
        Analyze website structure and generate insights
        
        Args:
            content (dict): Structured website content
            
        Returns:
            dict: Website analysis metrics
        """
        analysis = {
            'total_headings': len(content['headings']),
            'total_paragraphs': len(content['paragraphs']),
            'total_links': len(content['links']),
            'total_images': len(content['images']),
            'avg_paragraph_length': sum(len(p.split()) for p in content['paragraphs']) / len(content['paragraphs']) if content['paragraphs'] else 0,
            'images_with_alt': sum(1 for img in content['images'] if img['alt']),
            'content_structure': {
                'h1_count': sum(1 for h in content['headings'] if h.startswith('h1')),
                'h2_count': sum(1 for h in content['headings'] if h.startswith('h2')),
                'h3_count': sum(1 for h in content['headings'] if h.startswith('h3')),
            }
        }
        return analysis



class ContentAnalyzer:
    @staticmethod
    def extract_visual_content(content, content_type):
        """
        Extract visual content from PDF
        
        Args:
            content: PDF file
            content_type (str): 'PDF'
        
        Returns:
            list: Extracted visual content
        """
        visuals = []
        
        try:
            # PDF Image Extraction
            for page in content.pages:
                for image_file_object in page.images:
                    try:
                        img = Image.open(image_file_object)
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        visuals.append({
                            'image': img_byte_arr,
                            'text': pytesseract.image_to_string(img)
                        })
                    except Exception as e:
                        st.error(f"Image extraction error: {e}")
        
        except Exception as e:
            st.error(f"Visual content extraction error: {e}")
        
        return visuals

class FlashcardGenerator:
    @staticmethod
    def generate_flashcards(text, model, num_cards=5, answer_detail='brief'):
        """
        Generate educational flashcards from text content with flexible details
        
        Args:
            text (str): Content text
            model (ChatGoogleGenerativeAI): Gemini model
            num_cards (int): Number of flashcards
            answer_detail (str): Level of answer detail ('brief' or 'elaborate')
        
        Returns:
            list: Flashcard dictionaries
        """
        detail_instruction = "Provide brief, concise answers" if answer_detail == 'brief' else "Provide detailed, comprehensive explanations"
        
        prompt = f"""
        Generate {num_cards} educational flashcards from the following content. 
        Each flashcard should have:
        - Question
        - Answer ({detail_instruction})
        - Difficulty Level (Easy/Medium/Hard)
        
        Content: {text[:5000]}  # Limit to prevent excessive token usage
        
        Provide response in following JSON-like format:
        [
            {{
                "question": "Sample question about the content",
                "answer": "Explanation matching the detail level",
                "difficulty": "Easy"
            }}
        ]
        """
        
        try:
            response = model.invoke(prompt)
            # Use ast.literal_eval for safer parsing
            import ast
            flashcards = ast.literal_eval(response.content)
            return flashcards
        except Exception as e:
            st.error(f"Flashcard generation error: {e}")
            return []
    
class QuizGenerator:
    @staticmethod
    def generate_quiz(text, model, num_questions=5):
        """
        Generate a multiple-choice quiz from content with analysis
        
        Args:
            text (str): Content text
            model (ChatGoogleGenerativeAI): Gemini model
            num_questions (int): Number of quiz questions
        
        Returns:
            list: Quiz question dictionaries
        """
        prompt = f"""
        Generate {num_questions} multiple-choice quiz questions from the following content. 
        Each question should have:
        - Question text
        - 4 answer options
        - Correct answer index
        - Explanation for the correct answer
        
        Content: {text[:5000]}  # Limit to prevent excessive token usage
        
        Provide response in following JSON-like format:
        [
            {{
                "question": "Sample question about the content",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_index": 0,
                "explanation": "Detailed explanation of why this answer is correct"
            }}
        ]
        """
        
        try:
            response = model.invoke(prompt)
            # Use ast.literal_eval for safer parsing
            import ast
            quiz_questions = ast.literal_eval(response.content)
            return quiz_questions
        except Exception as e:
            st.error(f"Quiz generation error: {e}")
            return []
    @staticmethod

    def analyze_quiz_performance(quiz_questions, user_answers):
        """
        Analyze quiz performance and provide detailed feedback.

        Args:
            quiz_questions (list): Generated quiz questions.
            user_answers (list): User's selected answers.

        Returns:
        dict: Quiz performance analysis.
        """
        performance = {
        'total_questions': len(quiz_questions),
        'correct_answers': 0,
        'incorrect_answers': [],
        'detailed_feedback': []
    }
    
        for i, (question, user_answer) in enumerate(zip(quiz_questions, user_answers), 1):
            correct_answer = question['options'][question['correct_index']]
        if user_answer == correct_answer:
            performance['correct_answers'] += 1
        else:
            performance['incorrect_answers'].append({
                'question_number': i,
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'explanation': question.get('explanation', 'No explanation available')
            })
    
        performance['score_percentage'] = (performance['correct_answers'] / performance['total_questions']) * 100
    
        return performance
class DocumentProcessor:
    @staticmethod
    def process_pdf(pdf_docs):
        """
        Process PDF documents and extract text
        
        Args:
            pdf_docs (list): List of uploaded PDF files
        
        Returns:
            tuple: Raw text and text chunks
        """
        raw_text = ""
        
        try:
            # Concatenate text from multiple PDFs
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            text_chunks = text_splitter.split_text(raw_text)
            
            return raw_text, text_chunks
        
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return "", []

class EnhancedSummaryGenerator:
    @staticmethod
    def generate_content_summary(content, metadata, model, detail_level='Balanced', content_type='text'):
        """
        Generate comprehensive content summary with different detail levels
        
        Args:
            content (str): Main content text
            metadata (dict): Additional content metadata
            model: Language model for generation
            detail_level (str): Detail level (Detailed/Balanced/Brief)
            content_type (str): Type of content (text/pdf/website/video)
            
        Returns:
            dict: Structured summary information
        """
        if detail_level == 'Detailed':
            prompt = f"""
            Provide an extremely detailed and comprehensive analysis of the following {content_type} content.
            Include these sections in your analysis:

            1. Executive Summary (2-3 paragraphs)
               - Main thesis or central message
               - Key findings and conclusions
               - Critical insights

            2. Detailed Content Analysis
               - Major themes and concepts with in-depth explanation
               - Supporting evidence and examples
               - Technical details and terminology explained
               - Critical analysis of arguments presented
               - Relationships between different ideas
               
            3. Structure and Organization
               - Content flow and logical progression
               - Key sections and their interconnections
               {'- Timestamps for major topics and discussions' if content_type == 'video' else ''}
               {'- Website navigation and information architecture' if content_type == 'website' else ''}
               
            4. Supporting Elements
               - Statistics and data presented
               - Visual elements and their significance
               - Citations and references
               - Expert opinions or testimonials
               
            5. Context and Background
               - Historical or theoretical background
               - Industry or domain context
               - Related developments or research
               
            6. Practical Applications
               - Real-world implications
               - Actionable insights
               - Practical recommendations
               
            7. Critical Evaluation
               - Strengths and limitations
               - Potential biases or gaps
               - Alternative viewpoints
               - Quality of evidence
               
            8. Future Implications
               - Trends and predictions
               - Areas for further exploration
               - Potential developments

            {content_type.capitalize()} Title/Source: {metadata.get('title', 'Unknown')}
            {'Duration: ' + metadata.get('duration', 'Unknown') if content_type == 'video' else ''}
            {'URL: ' + metadata.get('url', 'Unknown') if content_type == 'website' else ''}
            
            Content:
            {content[:7000]}
            
            Provide a thorough, academic-style analysis with clear section headings and detailed explanations.
            """
            
        elif detail_level == 'Balanced':
            prompt = f"""
            Provide a well-balanced analysis of the following {content_type} content, maintaining detail while ensuring clarity.
            Include these sections:

            1. Overview (1-2 paragraphs)
               - Main message and purpose
               - Key findings
               
            2. Core Content Analysis
               - Principal themes and concepts
               - Key arguments and evidence
               - Important technical details
               {'- Timeline of major points discussed' if content_type == 'video' else ''}
               
            3. Structure and Organization
               - Content flow
               - Major sections
               {'- Key timestamps and topics' if content_type == 'video' else ''}
               {'- Website sections and navigation' if content_type == 'website' else ''}
               
            4. Key Insights
               - Main takeaways
               - Practical applications
               - Notable findings
               
            5. Supporting Elements
               - Important data and statistics
               - Significant examples
               - Expert insights
               
            6. Evaluation
               - Strengths and limitations
               - Quality of content
               - Areas for improvement

            {content_type.capitalize()} Title/Source: {metadata.get('title', 'Unknown')}
            {'Duration: ' + metadata.get('duration', 'Unknown') if content_type == 'video' else ''}
            {'URL: ' + metadata.get('url', 'Unknown') if content_type == 'website' else ''}
            
            Content:
            {content[:6000]}
            
            Balance depth and accessibility in your analysis.
            """
            
        else:  # Brief
            prompt = f"""
            Provide a concise but informative summary of the following {content_type} content:

            1. Quick Overview
               - Core message
               - Main purpose
               {'- Video timeline' if content_type == 'video' else ''}
               
            2. Key Points
               - Major themes
               - Essential arguments
               - Critical findings
               
            3. Main Takeaways
               - Principal conclusions
               - Practical implications
               - Action items
               
            {content_type.capitalize()} Title/Source: {metadata.get('title', 'Unknown')}
            {'Duration: ' + metadata.get('duration', 'Unknown') if content_type == 'video' else ''}
            {'URL: ' + metadata.get('url', 'Unknown') if content_type == 'website' else ''}
            
            Content:
            {content[:5000]}
            
            Focus on essential information while maintaining clarity and usefulness.
            """
        
        try:
            response = model.invoke(prompt)
            
            # Create structured summary output
            summary_info = {
                'metadata': metadata,
                'content_type': content_type,
                'detail_level': detail_level,
                'summary': response.content,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return summary_info
        except Exception as e:
            st.error(f"Summary generation error: {e}")
            return None

    @staticmethod
    def display_summary(summary_info, st):
        """
        Display formatted summary in Streamlit
        
        Args:
            summary_info (dict): Generated summary information
            st: Streamlit instance
        """
        try:
            # Display metadata
            st.subheader("Content Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Title:** {summary_info['metadata'].get('title', 'Unknown')}")
                st.write(f"**Type:** {summary_info['content_type'].capitalize()}")
                
            with col2:
                if summary_info['content_type'] == 'video':
                    st.write(f"**Duration:** {summary_info['metadata'].get('duration', 'Unknown')}")
                    st.write(f"**Views:** {summary_info['metadata'].get('views', 'Unknown')}")
                elif summary_info['content_type'] == 'website':
                    st.write(f"**URL:** {summary_info['metadata'].get('url', 'Unknown')}")
                elif summary_info['content_type'] == 'pdf':
                    st.write(f"**Pages:** {summary_info['metadata'].get('pages', 'Unknown')}")
                    
            # Display summary with expandable sections
            st.subheader("Summary Analysis")
            
            # Split summary into sections
            sections = summary_info['summary'].split('\n\n')
            
            for section in sections:
                if section.strip():
                    # Check if it's a section header
                    if section.startswith(('#', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                        st.markdown(f"### {section.split('.')[0].strip()}")
                        st.write(section.split('.', 1)[1].strip() if '.' in section else section)
                    else:
                        st.write(section.strip())
                        
            # Add timestamp markers for video content
            if summary_info['content_type'] == 'video' and 'timestamps' in summary_info['metadata']:
                st.subheader("Timeline Markers")
                for timestamp in summary_info['metadata']['timestamps']:
                    st.write(f"**[{timestamp['time']}]** {timestamp['topic']}")
                    
        except Exception as e:
            st.error(f"Error displaying summary: {e}")

def process_youtube_content(url, youtube_processor, model, detail_level):
    """Process YouTube video content"""
    video_info = youtube_processor.extract_video_info(url)
    if not video_info:
        return None
        
    transcript = youtube_processor.get_transcript(url)
    
    # Extract and format transcript text with timestamps
    transcript_text = ""
    timestamps = []
    
    if transcript:
        current_topic = ""
        topic_start_time = 0
        
        for i, segment in enumerate(transcript):
            start_time = segment.get('start')
            formatted_time = youtube_processor.format_timestamp(start_time) if start_time else "N/A"
            text = segment.get('text', '')
            transcript_text += f"[{formatted_time}] {text}\n"

            
            # Detect topic changes (simplified example - you might want to use more sophisticated topic detection)
            if i > 0 and i % 10 == 0:  # Every 10 segments, assume potential topic change
                timestamps.append({
                    'time': youtube_processor.format_timestamp(topic_start_time),
                    'topic': current_topic or f"Topic {len(timestamps) + 1}"
                })
                topic_start_time = segment['start']
    
    # Combine video information
    metadata = {
        'title': video_info['title'],
        'duration': video_info['duration'],
        'channel': video_info['channel'],
        'views': video_info['view_count'],
        'upload_date': video_info['upload_date'],
        'timestamps': timestamps
    }
    
    # Generate comprehensive summary
    content = f"{video_info['description']}\n\nTranscript:\n{transcript_text}"
    return EnhancedSummaryGenerator.generate_content_summary(
        content, metadata, model, detail_level, 'video'
    )

def process_website_content(url, website_processor, model, detail_level):
    """Process website content"""
    raw_text, structured_content = website_processor.extract_website_content(url)
    if not raw_text:
        return None
        
    metadata = {
        'title': structured_content.get('title', 'Unknown'),
        'url': url,
        'headings': len(structured_content.get('headings', [])),
        'paragraphs': len(structured_content.get('paragraphs', [])),
        'links': len(structured_content.get('links', [])),
        'images': len(structured_content.get('images', []))
    }
    
    return EnhancedSummaryGenerator.generate_content_summary(
        raw_text, metadata, model, detail_level, 'website'
    )

def process_pdf_content(pdf_docs, document_processor, model, detail_level):
    """Process PDF content"""
    raw_text, text_chunks = document_processor.process_pdf(pdf_docs)
    if not raw_text:
        return None
        
    metadata = {
        'title': pdf_docs[0].name,
        'pages': len(PdfReader(pdf_docs[0]).pages),
        'file_count': len(pdf_docs)
    }
    
    return EnhancedSummaryGenerator.generate_content_summary(
        raw_text, metadata, model, detail_level, 'pdf'
    )
class YouTubeProcessor:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }

    def extract_video_info(self, url):
        """
        Extract comprehensive information about a YouTube video
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Video information including title, description, duration, etc.
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                video_info = ydl.extract_info(url, download=False)
                
                # Format duration
                duration_sec = video_info.get('duration', 0)
                duration = str(timedelta(seconds=duration_sec))
                
                processed_info = {
                    'title': video_info.get('title', 'Unknown'),
                    'description': video_info.get('description', ''),
                    'duration': duration,
                    'view_count': video_info.get('view_count', 0),
                    'like_count': video_info.get('like_count', 0),
                    'upload_date': video_info.get('upload_date', ''),
                    'channel': video_info.get('uploader', 'Unknown'),
                    'tags': video_info.get('tags', []),
                    'categories': video_info.get('categories', []),
                    'subtitles': video_info.get('subtitles', {}),
                    'automatic_captions': video_info.get('automatic_captions', {})
                }
                
                return processed_info
        except Exception as e:
            st.error(f"Error extracting video info: {e}")
            return None

    def get_transcript(self, url):
        """
        Get video transcript using yt-dlp
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            list: List of transcript segments with timestamps
        """
        try:
            # Configure yt-dlp to download subtitles
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitlesformat': 'srt',
                'skip_download': True,
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                video_info = ydl.extract_info(url, download=False)
                
                # Try to get manual subtitles first, then fall back to auto-generated
                subtitles = video_info.get('subtitles', {})
                auto_subs = video_info.get('automatic_captions', {})
                
                # Prefer English subtitles
                if 'en' in subtitles:
                    return subtitles['en']
                elif 'en' in auto_subs:
                    return auto_subs['en']
                
                return None
        except Exception as e:
            st.error(f"Error getting transcript: {e}")
            return None

    @staticmethod
    def format_timestamp(seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds))) 

def main():
    st.set_page_config(page_title="Advanced Document Analyzer", page_icon="📄", layout="wide")
    st.title("📄 Advanced Document, YouTube & Website Analyzer")

    # Load environment variables and configure API
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize models
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Options")
        mode = st.selectbox(
            "Select Mode", 
            ["Summarize", "Search", "Learn"]
        )
        
        content_type = st.selectbox(
            "Select Content Type", 
            ["PDF", "Website", "YouTube Video"]
        )
        
        detail_level = st.radio(
            "Analysis Detail Level", 
            ["Brief", "Balanced", "Detailed"],
            index=1
        )
        
        # Additional options based on mode
        if mode == "Summarize":
            st.subheader("Summary Options")
            include_metadata = st.checkbox("Include Metadata Analysis", value=True)
            include_visuals = st.checkbox("Include Visual Analysis", value=True)
            
            use_custom_length = st.checkbox("Limit Response Length")
            max_length = None
            if use_custom_length:
                max_length = st.number_input(
                    "Maximum Characters", 
                    min_value=50, 
                    max_value=1000, 
                    value=500, 
                    step=50
                )
        
        elif mode == "Learn":
            st.subheader("Learning Options")
            num_flashcards = st.slider("Number of Flashcards", 5, 20, 10)
            num_quiz_questions = st.slider("Number of Quiz Questions", 5, 20, 10)
            answer_detail = st.radio("Answer Detail", ["Brief", "Elaborate"])

    # Content Type Processing
    if content_type == "YouTube Video":
        st.subheader("YouTube Video Analysis")
        video_url = st.text_input("Enter YouTube Video URL")
        
        if video_url:
            youtube_processor = YouTubeProcessor()
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Video Summary",
                "Transcript Analysis",
                "Learning Materials",
                "Visual Analysis"
            ])
            
            with tab1:
                if st.button("Generate Video Summary"):
                    with st.spinner("Analyzing video..."):
                        summary_info = process_youtube_content(
                            video_url,
                            youtube_processor,
                            model,
                            detail_level
                        )
                        
                        if summary_info:
                            EnhancedSummaryGenerator.display_summary(summary_info, st)
            
            with tab2:
                if st.button("Analyze Transcript"):
                    with st.spinner("Processing transcript..."):
                        transcript = youtube_processor.get_transcript(video_url)
                        if transcript:
                            # Display transcript with interactive features
                            st.subheader("Interactive Transcript")
                            
                            # Add search functionality
                            search_term = st.text_input("Search in transcript:")
                            
                            # Display transcript with timestamps and search highlighting
                            for segment in transcript:
                                timestamp = youtube_processor.format_timestamp(segment['start'])
                                text = segment['text']
                                
                                if not search_term or search_term.lower() in text.lower():
                                    col1, col2 = st.columns([1, 4])
                                    with col1:
                                        st.code(timestamp)
                                    with col2:
                                        if search_term:
                                            highlighted_text = text.replace(
                                                search_term,
                                                f"**{search_term}**"
                                            )
                                            st.markdown(highlighted_text)
                                        else:
                                            st.write(text)
            with tab3:
                if st.button("Generate Learning Materials"):
                    with st.spinner("Creating educational content..."):
                        video_info = youtube_processor.extract_video_info(video_url)
                        if video_info:
                # Generate flashcards
                            st.subheader("Flashcards")
                            flashcards = FlashcardGenerator.generate_flashcards(
                            video_info['description'],
                            model,
                            num_cards=num_flashcards,
                            answer_detail=answer_detail
                )
                
                            for i, card in enumerate(flashcards, 1):
                                with st.expander(f"Flashcard {i} - {card['difficulty']}"):
                                    st.write(f"**Q:** {card['question']}")
                                    st.write(f"**A:** {card['answer']}")
                
                # Generate quiz
                            st.subheader("Practice Quiz")
                            quiz = QuizGenerator.generate_quiz(
                                video_info['description'],
                                model,
                                num_questions=num_quiz_questions
                )
                
                            if quiz:    
                    # Store quiz questions in session state if not already stored
                                if 'quiz_questions' not in st.session_state:
                                    st.session_state.quiz_questions = quiz
                    
                    # Initialize submit_status if not in session state
                                if 'submit_status' not in st.session_state:
                                    st.session_state.submit_status = False
                    
                    # Create quiz form
                                with st.form(key="quiz_form"):
                                    user_answers = []
                                    for i, q in enumerate(st.session_state.quiz_questions, 1):
                                        st.write(f"**Question {i}:** {q['question']}")
                                        answer = st.radio(
                                "Select your answer:",
                                q['options'],
                                key=f"q{i}"
                            )
                                    user_answers.append(answer)
                        
                        # Add submit button
                                    submit_button = st.form_submit_button("Submit Quiz")
                        
                                    if submit_button:
                                        st.session_state.submit_status = True
                                        st.session_state.user_answers = user_answers
                    
                    # Handle form submission outside the form
                                if st.session_state.submit_status:
                        # Analyze quiz performance
                                    performance = QuizGenerator.analyze_quiz_performance(
                                        st.session_state.quiz_questions,
                                        st.session_state.user_answers
                        )
                        
                        # Display overall performance
                                    st.success(f"Your Score: {performance['correct_answers']}/{performance['total_questions']} "
                                            f"({performance['score_percentage']:.2f}%)")
                        
                        # Detailed performance breakdown
                                    if performance['incorrect_answers']:
                                        st.subheader("Detailed Performance")
                                        for incorrect in performance['incorrect_answers']:
                                            with st.expander(f"Question {incorrect['question_number']} - Incorrect"):
                                                st.write(f"**Question:** {incorrect['question']}")
                                                st.write(f"**Your Answer:** {incorrect['user_answer']}")
                                                st.write(f"**Correct Answer:** {incorrect['correct_answer']}")
                                                st.write(f"**Explanation:** {incorrect['explanation']}")
                        
                                    st.balloons()
                        
                        # Reset submit status for next attempt
                                    if st.button("Take Quiz Again"):
                                        st.session_state.submit_status = False
                                        st.experimental_rerun()

            with tab4:
                if st.button("Analyze Visual Content"):
                    with st.spinner("Analyzing visual elements..."):
            # Add visual content analysis here
                        st.info("Visual content analysis will be added in a future update")                         
    elif content_type == "Website":
        st.subheader("Website Analysis")
        url = st.text_input("Enter Website URL", placeholder="https://example.com")
        
        if url:
            if not WebsiteProcessor.is_valid_url(url):
                st.error("Please enter a valid URL")
            else:
                website_processor = WebsiteProcessor()
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Content Summary",
                    "Structure Analysis",
                    "Learning Materials",
                    "Visual Elements"
                ])
                
                with tab1:
                    if st.button("Generate Website Summary"):
                        with st.spinner("Analyzing website content..."):
                            summary_info = process_website_content(
                                url,
                                website_processor,
                                model,
                                detail_level
                            )
                            
                            if summary_info:
                                EnhancedSummaryGenerator.display_summary(summary_info, st)
                
                with tab2:
                    if st.button("Analyze Website Structure"):
                        with st.spinner("Analyzing structure..."):
                            raw_text, content = website_processor.extract_website_content(url)
                            analysis = website_processor.analyze_website_structure(content)
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Headings", analysis['total_headings'])
                                st.metric("Total Paragraphs", analysis['total_paragraphs'])
                            with col2:
                                st.metric("Total Links", analysis['total_links'])
                                st.metric("Total Images", analysis['total_images'])
                            with col3:
                                st.metric("Images with Alt Text", analysis['images_with_alt'])
                                st.metric("Avg Paragraph Length", 
                                        f"{analysis['avg_paragraph_length']:.1f} words")
                            
                            # Display content hierarchy
                            st.subheader("Content Hierarchy")
                            st.json(analysis['content_structure'])
                
                with tab3:
                    if st.button("Generate Learning Materials"):
                        with st.spinner("Creating educational content..."):
                            raw_text, _ = website_processor.extract_website_content(url)
                            
                            # Generate flashcards
                            flashcards = FlashcardGenerator.generate_flashcards(
                                raw_text,
                                model,
                                num_cards=num_flashcards,
                                answer_detail=answer_detail
                            )
                            
                            st.subheader("Flashcards")
                            for i, card in enumerate(flashcards, 1):
                                with st.expander(f"Flashcard {i} - {card['difficulty']}"):
                                    st.write(f"**Q:** {card['question']}")
                                    st.write(f"**A:** {card['answer']}")
                            
                            # Generate quiz
                            st.subheader("Practice Quiz")
                            quiz = QuizGenerator.generate_quiz(
                                raw_text,
                                model,
                                num_questions=num_quiz_questions
                            )
                            
                            if quiz:
                                with st.form("website_quiz"):
                                    user_answers = []
                                    for i, q in enumerate(quiz, 1):
                                        st.write(f"**Question {i}:** {q['question']}")
                                        answer = st.radio(
                                            "Select your answer:",
                                            q['options'],
                                            key=f"wq{i}"
                                        )
                                        user_answers.append(answer)
                                    
                                    if st.form_submit_button("Submit Quiz"):
                                        performance = QuizGenerator.analyze_quiz_performance(
                                            quiz,
                                            user_answers
                                        )
                                        st.success(
                                            f"Score: {performance['score_percentage']:.1f}%"
                                        )
                                        
                                        if performance['incorrect_answers']:
                                            st.subheader("Review Incorrect Answers")
                                            for wrong in performance['incorrect_answers']:
                                                with st.expander(f"Question {wrong['question_number']}"):
                                                    st.write(f"**Question:** {wrong['question']}")
                                                    st.write(f"**Your Answer:** {wrong['user_answer']}")
                                                    st.write(f"**Correct Answer:** {wrong['correct_answer']}")
                                                    st.write(f"**Explanation:** {wrong['explanation']}")
                
                with tab4:
                    if st.button("Analyze Visual Elements"):
                        with st.spinner("Analyzing visual elements..."):
                            _, content = website_processor.extract_website_content(url)
                            
                            if content['images']:
                                st.write(f"Found {len(content['images'])} images")
                                for i, img in enumerate(content['images'], 1):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        try:
                                            full_url = urllib.parse.urljoin(url, img['src'])
                                            st.image(full_url, caption=f"Image {i}")
                                        except:
                                            st.error(f"Could not load image {i}")
                                    with col2:
                                        st.write("**Alt Text:**", img['alt'] or "No alt text")
                                        st.write("**Source:**", img['src'])
                            else:
                                st.info("No images found on the webpage")
    elif content_type == "PDF":
        st.subheader("PDF Analysis")
        pdf_docs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
        
        if pdf_docs:
            document_processor = DocumentProcessor()
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "PDF Summary",
                "Content Analysis",
                "Learning Materials",
                "Visual Elements"
            ])
            
            with tab1:
                if st.button("Generate PDF Summary"):
                    with st.spinner("Analyzing PDF content..."):
                        summary_info = process_pdf_content(
                            pdf_docs,
                            document_processor,
                            model,
                            detail_level
                        )
                        
                        if summary_info:
                            EnhancedSummaryGenerator.display_summary(summary_info, st)
            
            with tab2:
                if st.button("Analyze PDF Structure"):
                    with st.spinner("Analyzing document structure..."):
                        raw_text, text_chunks = document_processor.process_pdf(pdf_docs)
                        if raw_text:
                            # Display document metrics
                            total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
                            total_chars = len(raw_text)
                            total_words = len(raw_text.split())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Pages", total_pages)
                            with col2:
                                st.metric("Total Words", total_words)
                            with col3:
                                st.metric("Total Characters", total_chars)
                            
                            # Display content overview
                            st.subheader("Content Overview")
                            chunks_sample = text_chunks[:5]  # Show first 5 chunks
                            for i, chunk in enumerate(chunks_sample, 1):
                                with st.expander(f"Content Section {i}"):
                                    st.write(chunk)
            
            with tab3:
                if st.button("Generate Learning Materials"):
                    with st.spinner("Creating educational content..."):
                        raw_text, _ = document_processor.process_pdf(pdf_docs)
                        
                        if raw_text:
                            # Generate flashcards
                            st.subheader("Flashcards")
                            flashcards = FlashcardGenerator.generate_flashcards(
                                raw_text,
                                model,
                                num_cards=num_flashcards,
                                answer_detail=answer_detail
                            )
                            
                            for i, card in enumerate(flashcards, 1):
                                with st.expander(f"Flashcard {i} - {card['difficulty']}"):
                                    st.write(f"**Q:** {card['question']}")
                                    st.write(f"**A:** {card['answer']}")
                            
                            # Generate quiz
                            st.subheader("Practice Quiz")
                            quiz = QuizGenerator.generate_quiz(
                                raw_text,
                                model,
                                num_questions=num_quiz_questions
                            )
                            
                            if quiz:
                                with st.form("pdf_quiz"):
                                    user_answers = []
                                    for i, q in enumerate(quiz, 1):
                                        st.write(f"**Question {i}:** {q['question']}")
                                        answer = st.radio(
                                            "Select your answer:",
                                            q['options'],
                                            key=f"pq{i}"
                                        )
                                        user_answers.append(answer)
                                    
                                    if st.form_submit_button("Submit Quiz"):
                                        performance = QuizGenerator.analyze_quiz_performance(
                                            quiz,
                                            user_answers
                                        )
                                        st.success(
                                            f"Score: {performance['score_percentage']:.1f}%"
                                        )
                                        
                                        if performance['incorrect_answers']:
                                            st.subheader("Review Incorrect Answers")
                                            for wrong in performance['incorrect_answers']:
                                                with st.expander(f"Question {wrong['question_number']}"):
                                                    st.write(f"**Question:** {wrong['question']}")
                                                    st.write(f"**Your Answer:** {wrong['user_answer']}")
                                                    st.write(f"**Correct Answer:** {wrong['correct_answer']}")
                                                    st.write(f"**Explanation:** {wrong['explanation']}")
            
            with tab4:
                if st.button("Analyze Visual Elements"):
                    with st.spinner("Analyzing visual elements..."):
                        for pdf in pdf_docs:
                            pdf_reader = PdfReader(pdf)
                            visuals = ContentAnalyzer.extract_visual_content(
                                pdf_reader, 
                                'PDF'
                            )
                            
                            if visuals:
                                st.subheader(f"Visual Elements from {pdf.name}")
                                for i, visual in enumerate(visuals, 1):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(visual['image'], caption=f"Image {i}")
                                    with col2:
                                        if visual.get('text'):
                                            st.write("**Extracted Text:**")
                                            st.write(visual['text'])
                            else:
                                st.info(f"No images found in {pdf.name}")


if __name__ == "__main__":
    main()