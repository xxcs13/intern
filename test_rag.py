#!/usr/bin/env python3
"""
Test script for the RAG Q&A system
"""
import os
from rag import graph

def test_rag_system():
    """Test the RAG system with sample files."""
    
    # Check if required files exist
    required_files = [
        "./tsmc_2024_yearly report.pdf",
        "./無備忘錄版_digitimes_2025年全球AI伺服器出貨將達181萬台　高階機種採購不再集中於四大CSP.pptx",
        "./excel_FS-Consolidated_1Q25.xls"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("Please ensure these files exist in the current directory.")
        return False
    
    # Test questions
    test_questions = [
        "What is the global AI server market size in 2025?",
        "What are the key highlights from TSMC's 2024 yearly report?",
        "What information is available in the Excel financial report?"
    ]
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {question}")
        print('='*60)
        
        try:
            state = {
                "pdf_path": "./tsmc_2024_yearly report.pdf",
                "pptx_path": "./無備忘錄版_digitimes_2025年全球AI伺服器出貨將達181萬台　高階機種採購不再集中於四大CSP.pptx",
                "xls_path": "./excel_FS-Consolidated_1Q25.xls",
                "question": question
            }
            
            result = graph.invoke(state)
            print(f"Answer: {result['answer']}")
            print(f"✅ Test {i} completed successfully!")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print("🎉 All tests completed! Check 'rag_qa_log.csv' for logged Q&A sessions.")
    print('='*60)
    return True

if __name__ == "__main__":
    test_rag_system()
