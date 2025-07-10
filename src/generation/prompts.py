"""
Prompt templates for the RAG pipeline.
Contains structured prompts for different use cases.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts available."""
    QA = "qa"
    CONVERSATIONAL = "conversational"
    SUMMARIZATION = "summarization"
    SEARCH = "search"
    CONTEXT_ENHANCEMENT = "context_enhancement"


@dataclass
class PromptTemplate:
    """Template for prompts."""
    name: str
    template: str
    system_message: Optional[str] = None
    variables: List[str] = None
    description: str = ""

    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")


class PromptManager:
    """Manager for prompt templates."""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize default prompt templates."""
        return {
            "basic_qa": PromptTemplate(
                name="basic_qa",
                template="""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:""",
                system_message="You are a helpful AI assistant that answers questions based on provided context. Be accurate, concise, and honest about what you know.",
                variables=["context", "question"],
                description="Basic question-answering prompt"
            ),

            "conversational_qa": PromptTemplate(
                name="conversational_qa",
                template="""You are having a conversation with a user. Use the following context to inform your response, but maintain a natural conversational tone.

Context:
{context}

Chat History:
{chat_history}

Current Question: {question}

Response:""",
                system_message="You are a helpful AI assistant engaged in a conversation. Use the provided context to give accurate answers while maintaining a natural, friendly tone.",
                variables=["context", "chat_history", "question"],
                description="Conversational question-answering with chat history"
            ),

            "qa_with_sources": PromptTemplate(
                name="qa_with_sources",
                template="""Answer the question based on the provided context. Include source references in your answer.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the context
- Include relevant source references when making claims
- If information is not available in the context, state this clearly

Answer:""",
                system_message="You are a helpful AI assistant that provides detailed answers with proper source attribution. Always cite your sources when making claims.",
                variables=["context", "question"],
                description="QA with source attribution"
            ),

            "query_enhancement": PromptTemplate(
                name="query_enhancement",
                template="""Given the user's question and chat history, generate 2-3 enhanced search queries that would help find relevant information. The enhanced queries should:
- Expand on the original question with relevant context
- Include synonyms and related terms
- Be specific enough to find targeted information

Original Question: {question}

Chat History:
{chat_history}

Enhanced Search Queries:""",
                system_message="You are a search query enhancement specialist. Generate improved search queries that will help find the most relevant information.",
                variables=["question", "chat_history"],
                description="Enhance user queries for better retrieval"
            ),

            "context_summarization": PromptTemplate(
                name="context_summarization",
                template="""Summarize the following context to focus on information most relevant to the user's question. Keep important details while removing less relevant information.

Question: {question}

Context to Summarize:
{context}

Relevant Summary:""",
                system_message="You are a context summarization specialist. Extract and summarize the most relevant information for answering the user's question.",
                variables=["question", "context"],
                description="Summarize context for focused answers"
            ),

            "multi_turn_conversation": PromptTemplate(
                name="multi_turn_conversation",
                template="""You are engaged in a multi-turn conversation. Consider the entire conversation history and the current context to provide a helpful response.

Context:
{context}

Full Conversation History:
{full_history}

Current User Message: {current_message}

Response:""",
                system_message="You are a helpful AI assistant engaged in a multi-turn conversation. Use context and conversation history to provide relevant, coherent responses.",
                variables=["context", "full_history", "current_message"],
                description="Multi-turn conversation with full history"
            ),

            "safety_check": PromptTemplate(
                name="safety_check",
                template="""Analyze the following response for potential safety issues:

Response to Check:
{response}

Check for:
- Harmful or inappropriate content
- Potential misinformation
- Privacy violations
- Offensive language

Assessment:""",
                system_message="You are a content safety analyst. Evaluate responses for potential safety issues and provide recommendations.",
                variables=["response"],
                description="Safety analysis for generated responses"
            ),

            "document_classification": PromptTemplate(
                name="document_classification",
                template="""Classify the following document and extract key metadata:

Document:
{document}

Please provide:
1. Document type/category
2. Main topics covered
3. Key entities mentioned
4. Document summary (2-3 sentences)

Classification:""",
                system_message="You are a document classification specialist. Analyze documents and extract relevant metadata for indexing and retrieval.",
                variables=["document"],
                description="Document classification and metadata extraction"
            ),

            "fact_check": PromptTemplate(
                name="fact_check",
                template="""Fact-check the following statement against the provided context:

Statement to Check: {statement}

Context:
{context}

Analysis:
- Is the statement supported by the context?
- What evidence supports or contradicts it?
- Are there any nuances or caveats?

Fact Check Result:""",
                system_message="You are a fact-checking specialist. Carefully analyze statements against provided context and identify any inaccuracies or missing nuances.",
                variables=["statement", "context"],
                description="Fact-checking statements against context"
            ),

            "explanation": PromptTemplate(
                name="explanation",
                template="""Explain the following concept in detail using the provided context. Make your explanation clear and accessible.

Concept to Explain: {concept}

Context:
{context}

Target Audience: {audience}

Explanation:""",
                system_message="You are an expert educator. Provide clear, comprehensive explanations tailored to your audience's level of understanding.",
                variables=["concept", "context", "audience"],
                description="Detailed explanations of concepts"
            ),

            "research_synthesis": PromptTemplate(
                name="research_synthesis",
                template="""Synthesize information from multiple sources to provide a comprehensive answer to the research question.

Research Question: {question}

Sources:
{sources}

Instructions:
- Integrate information from all sources
- Identify agreements and disagreements between sources
- Provide a balanced, well-supported answer
- Note any limitations in the available information

Synthesis:""",
                system_message="You are a research synthesis specialist. Integrate information from multiple sources to provide comprehensive, balanced answers to research questions.",
                variables=["question", "sources"],
                description="Synthesize information from multiple sources"
            )
        }

    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())

    def get_templates_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """Get templates by type."""
        # Simple implementation, later I'll tag templates
        type_mapping = {
            PromptType.QA: ["basic_qa", "qa_with_sources"],
            PromptType.CONVERSATIONAL: ["conversational_qa", "multi_turn_conversation"],
            PromptType.SUMMARIZATION: ["context_summarization", "research_synthesis"],
            PromptType.SEARCH: ["query_enhancement"],
            PromptType.CONTEXT_ENHANCEMENT: ["context_summarization", "query_enhancement"]
        }

        template_names = type_mapping.get(prompt_type, [])
        return [self.templates[name] for name in template_names if name in self.templates]

    def add_template(self, template: PromptTemplate):
        """Add a new template."""
        self.templates[template.name] = template

    def update_template(self, name: str, template: PromptTemplate):
        """Update an existing template."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        self.templates[name] = template

    def remove_template(self, name: str):
        """Remove a template."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        del self.templates[name]


_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt_template(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    return get_prompt_manager().get_template(name)


def format_prompt(name: str, **kwargs) -> str:
    """Format a prompt template with variables."""
    template = get_prompt_template(name)
    return template.format(**kwargs)


def get_system_message(name: str) -> Optional[str]:
    """Get system message for a template."""
    template = get_prompt_template(name)
    return template.system_message


def format_context(documents: List[Dict[str, Any]], max_length: int = 4000) -> str:
    """Format documents as context for prompts."""
    formatted_docs = []
    total_length = 0

    for i, doc in enumerate(documents):
        content = doc.get("content", "")
        source = doc.get("source", f"Document {i+1}")

        doc_text = f"Source: {source}\nContent: {content}\n"

        if total_length + len(doc_text) > max_length:
            break

        formatted_docs.append(doc_text)
        total_length += len(doc_text)

    return "\n---\n".join(formatted_docs)


def format_chat_history(messages: List[Dict[str, str]], max_messages: int = 10) -> str:
    """Format chat history for prompts."""
    if not messages:
        return "No previous conversation."

    # Take only the last N messages
    recent_messages = messages[-max_messages:]

    formatted_history = []
    for msg in recent_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
        elif role == "system":
            formatted_history.append(f"System: {content}")

    return "\n".join(formatted_history)


def format_sources(documents: List[Dict[str, Any]]) -> str:
    """Format documents as numbered sources."""
    formatted_sources = []

    for i, doc in enumerate(documents, 1):
        source = doc.get("source", f"Source {i}")
        content = doc.get("content", "")

        formatted_sources.append(f"[{i}] {source}: {content}")

    return "\n\n".join(formatted_sources)