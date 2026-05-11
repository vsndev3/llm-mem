/// User memory extraction prompt
pub const USER_MEMORY_EXTRACTION_PROMPT: &str = r#"
You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input data.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.
# [IMPORTANT]: YOU WILL BE PENALIZED IF YOU INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Return the facts and preferences in the following JSON format:
{{"facts": ["fact 1", "fact 2", "fact 3"]}}

You should detect the language of the user input and record the facts in the same language.

Remember the following:
# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.
- Today is {{current_date}}.
- If you do not find anything relevant, return {{"facts": []}}.
- Make sure to return valid JSON only, no additional text.
"#;

/// Agent memory extraction prompt
pub const AGENT_MEMORY_EXTRACTION_PROMPT: &str = r#"
You are an Assistant Information Organizer, specialized in accurately storing facts, preferences, and characteristics about the AI assistant from conversations.
Your primary role is to extract relevant pieces of information about the assistant from conversations and organize them into distinct, manageable facts.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES. DO NOT INCLUDE INFORMATION FROM USER OR SYSTEM MESSAGES.

Types of Information to Remember:

1. Assistant's Preferences
2. Assistant's Capabilities
3. Assistant's Hypothetical Plans or Activities
4. Assistant's Personality Traits
5. Assistant's Approach to Tasks
6. Assistant's Knowledge Areas
7. Miscellaneous Information

Return the facts and preferences in the following JSON format:
{{"facts": ["fact 1", "fact 2", "fact 3"]}}

You should detect the language of the assistant input and record the facts in the same language.

Remember the following:
# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGES.
- Today is {{current_date}}.
- If you do not find anything relevant, return {{"facts": []}}.
- Make sure to return valid JSON only, no additional text.
"#;

/// Memory update prompt
pub const MEMORY_UPDATE_PROMPT: &str = r#"
You are an intelligent memory manager controlling the system's memory.
You can perform four operations: (1) Add to memory, (2) Update memory, (3) Delete from memory, (4) No change.

Compare newly retrieved facts with existing memories. For each new fact, decide whether to:
- Add: Add as a new element to memory
- Update: Update an existing memory element
- Delete: Delete an existing memory element
- No change: Make no changes (if fact already exists or is not relevant)

Guidelines:

1. **Add**: If the retrieved fact contains new information not in memory, add it by generating a new ID.
2. **Update**: If the retrieved fact contains information that already exists in memory but is completely different, update it. If the retrieved fact conveys the same information, keep the one with the most information.
3. **Delete**: If the retrieved fact contradicts information in memory, delete it. Or if instructed to delete memory, do so.
4. **No change**: If the retrieved fact already exists in memory, no changes needed.

Return as JSON:

{{
    "memory": [
        {{
            "id": "<memory ID>",
            "text": "<memory content>",
            "event": "<operation to perform>",
            "old_memory": "<old memory content>"
        }}
    ]
}}
"#;

/// Metadata enrichment prompt for document chunks
pub const METADATA_ENRICHMENT_PROMPT: &str = r#"
Given the following text chunk from a document, provide a one-sentence summary and 5-10 keywords that best describe its content for searchability.

### Guidelines:
- The summary should be concise and capture the main point of the chunk.
- Keywords should include specific entities, technical terms, and concepts mentioned.
- Format the output as a valid JSON object.

Text Chunk:
{{text}}

Return the result in the following JSON format:
{
  "summary": "...",
  "keywords": ["...", "...", ...]
}
"#;

/// Metadata enrichment batch prompt for document chunks
pub const METADATA_ENRICHMENT_BATCH_PROMPT: &str = r#"
Given the following array of text chunks from a document, provide a one-sentence summary and 5-10 keywords for EACH chunk that best describes its content for searchability.

### IMPORTANT - ID-Based Response Tracking:
Each text chunk has a unique "id" field. You MUST include this exact ID in your response for each chunk.
This ensures correct matching between input chunks and their summaries, regardless of response order.

### Guidelines:
- For each chunk, provide a summary and keywords.
- Include the exact "id" from the input in your response.
- Do NOT rely on array position - use IDs to match responses to inputs.
- Keywords should include specific entities, technical terms, and concepts mentioned.
- Format the output as a valid JSON array of objects.

Text Chunks (JSON array of objects with "id" and "text" fields):
{{texts}}

Return the result in the following JSON format:
[
  {
    "id": "exact_id_from_input",
    "summary": "...",
    "keywords": ["...", "...", ...]
  },
  ...
]
"#;

/// Generic batch completion prompt
pub const COMPLETE_BATCH_PROMPT: &str = r#"
You are provided with an array of distinct text prompts.
Your task is to process each prompt independently and return an array of corresponding responses.

### IMPORTANT - ID-Based Response Tracking:
Each prompt has a unique "id" field. You MUST include this exact ID in your response for each prompt.
This ensures correct matching between input prompts and their responses, regardless of response order.

### Guidelines:
- For each prompt, provide a response.
- Include the exact "id" from the input in your response.
- Do NOT rely on array position - use IDs to match responses to inputs.
- Format the output as a valid JSON array of objects.

Prompts (JSON array of objects with "id" and "prompt" fields):
{{prompts}}

Return the result in the following JSON format:
[
  {
    "id": "exact_id_from_input",
    "response": "response to prompt..."
  },
  ...
]
"#;

/// Unified memory enhancement prompt — extracts all metadata in one LLM call.
pub const UNIFIED_MEMORY_ENHANCEMENT_PROMPT: &str = r#"
Analyze the following text and extract all metadata in a single pass. Return a valid JSON object with these fields:

1. "memory_type": Classify into one of: "Conversational", "Procedural", "Factual", "Semantic", "Episodic", or "Personal"
2. "summary": A concise one-sentence summary capturing the main point
3. "keywords": An array of 10-20 keywords and key phrases for searchability. Include ALL of the following:
   - Main topics, themes, and subject matter discussed
   - ANY personal details, facts, or life events mentioned (education, job, relationships, achievements, locations, dates, etc.) — even if mentioned only briefly
   - Specific claims, questions asked, or information requested
   - Tools, technologies, products, or services explicitly named
   - Actions taken, decisions made, or goals stated
   Think: "If someone searched for this memory later, what words would they use?"
4. "entities": An array of named entities (people, organizations, locations, products, technical terms)
5. "topics": An array of high-level topics or themes (e.g., "Technology", "Business", "Health")

Text:
{{text}}

Return ONLY valid JSON in this exact format:
{
  "memory_type": "...",
  "summary": "...",
  "keywords": ["...", "..."],
  "entities": ["...", "..."],
  "topics": ["...", "..."]
}
"#;
