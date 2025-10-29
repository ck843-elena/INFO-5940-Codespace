# 💕RAG Document Chat💕 Instruactions
 

## API key & environment setup 

### 1. Environment setup
I use the  Fork button created a copy of the repo under my own GitHub account.

### 2. API Key
I set my API key in the devcontainer.json file to make it available inside the development environment.


## Interface layout description

RAG Document Chat is an intelligent document Q&A assistant. It enables you to upload your own documents (in PDF or TXT format), and then ask questions to the documents. The system will give accurate answers based on the content of the documents

### Left sidebar: Settings and Control Area

1. Settings Area

   1. Chunk Size slider 
 
      - Default value: 500 characters 
      - Adjustment range: 100-1000 characters 
      - Function: Control how large the document is divided into segments. Smaller chunks are more refined, while larger chunks retain more context 
 
   2. Chunk Overlap slider 
   
      - Default value: 50 characters 
      - Adjustment range: 0-200 characters 
      - Function: Control the number of overlapping characters between adjacent document blocks to prevent important information from being cut off at the block boundaries 
 
   3. Retrieved Chunks slider 
 
      - Default value: 5 
      - Adjustment range: 1 to 20 
      - Function: Set the number of relevant fragments to be retrieved from the document when answering questions 
 
   4. Chunking Strategy drop-down menu 
      - Option 1: Recursive (default) - Recursive partitioning, suitable for most documents 
      - Option 2: By Paragraph (\n\n) - Block by paragraph 
      - Option 3: PDF: one page = one chunk - PDF chunks by page 
      - Function: Select different document splitting methods

### Upload Document Area

1. Document upload area 

   1. Drag and drop area: Display "Drag and drop files here" 
file 
   2. Limit: 200MB per file • TXT, PDF 
   3. Browse files button: Click to browse local files 
Support simultaneous upload of multiple files 
 
2. Process Documents button 
 
   1. Location: Below the upload area 
   2. Status: Gray disabled status before uploading files 
   3. Function: Click to start processing the uploaded document 
      -The processing procedure will display: 
 
      - The file name being processed 
      - The process of attempting file encoding 
      - Document loading page count (PDF) 
      - The number of blocks 
      - Vectorized progress 
      - Final success hint
   
 
   2. Clear Conversation button 
 
      - Function: Clear all current conversation records 
      - Retained: Uploaded document libraries will not be deleted 
      - Purpose: Start asking questions again, but there is no need to re-upload the document 
 
   3. Reset All button 
 
      - Function: Completely reset the system 
Clear: Conversation records + document libraries + all temporary files 
      - Purpose: Start from scratch and re-upload a new document


## Detailed User Instructions 

Step 1: Upload the document

- Click the "Browse files" button or simply drag the file to the upload area 
- Select your TXT or PDF file (multiple choices are available) 
- Confirm that the file has been added to the upload list

Step 2: Process the document

- Click the "Process Documents" button 
- Wait for the system to process (detailed steps will be displayed) : 
   - File loading 
   - Document fragmentation 
   - Create vector embeddings 
   - Build a vector database 
 
   You can start asking questions after seeing the green success prompt

Step 3: Start asking questions
- Enter your question in the input box at the bottom 
- Press Enter to send 
- View the AI's responses based on the content of your document 
- Click "Sources" to view the source of the answer

Step 4: Keep the conversation going 
 
- Keep asking questions and the system will retain the context of the conversation 
- Click "Clear Conversation" when needed to clear the conversation record 
- When you need to change the document, click "Reset All" to start over

