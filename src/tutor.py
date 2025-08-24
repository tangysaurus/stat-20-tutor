from typing import Optional, List
from pydantic import Field, BaseModel
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class KnowledgeGraph:
    SYSTEM_PROMPT = (
        "# Knowledge Graph Instructions for GPT-4\n"
        "## 1. Overview\n"
        "You are a top-tier algorithm designed for extracting information in structured "
        "formats to build a knowledge graph for an educational learning platform.\n"
        "Try to capture as much information from the text as possible without "
        "sacrificing accuracy. Do not add any information that is not explicitly "
        "mentioned in the text.\n"
        "- **Nodes** represent entities and concepts.\n"
        "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
        "helpful for a student who is reviewing for an exam.\n"
        "## 2. Labeling Nodes\n"
        "- **Consistency**: Ensure you use available types for node labels.\n"
        "Ensure you use basic or elementary types for node labels.\n"
        "- For example, when you identify an entity representing a person, "
        "always label it as **'person'**. Avoid using more specific terms "
        "like 'mathematician' or 'scientist'."
        "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
        "names or human-readable identifiers found in the text.\n"
        "- **Relationships** represent connections between entities or concepts.\n"
        "Ensure consistency and generality in relationship types when constructing "
        "knowledge graphs. Instead of using specific and momentary types "
        "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
        "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
        "## 3. Coreference Resolution\n"
        "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
        "ensure consistency.\n"
        'If an entity, such as "John Doe", is mentioned multiple times in the text '
        'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
        "always use the most complete identifier for that entity throughout the "
        'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
        "Remember, the knowledge graph should be coherent and easily understandable, "
        "so maintaining consistency in entity references is crucial.\n"
        "## 4. Strict Compliance\n"
        "Adhere to the rules strictly. Non-compliance will result in termination."
    )

    USER_PROMPT = (
        "Ensure that the entities and concepts extracted are relevant material. They should not be teachers, course names, syllabus, or miscellaneous references."
        "Tip: Make sure to answer in the correct format and do "
        "not include any explanations. "
        "Use the given format to extract information from the "
        "following input: {input}"
    )

    def __init__(
        self, 
        uri: str, 
        username: str, 
        password: str, 
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        user_prompt: Optional[str] = USER_PROMPT
    ):
        """Initialize graph using user credentials"""
        self.uri = uri
        self.username = username
        self.password = password
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        self.graph = Neo4jGraph(
            url = self.uri,
            username = self.username,
            password = self.password,
            refresh_schema = False
        )

        self.prompt = self.create_prompt()
        self.chat_history = []

    def create_prompt(self):
        """Define prompt template"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt)
            ]
        )
    
    def build_graph(
        self, 
        llm, 
        documents, 
        batch_size = 32,
        max_workers = 6,
        sleep_between_batches = 0.0
        # nodes, 
        # relationships, 
        # node_properties
    ):
        """Build graph using specified entities & relationships"""
        xform = LLMGraphTransformer(
            llm = llm, 
            # allowed_nodes = nodes,
            # allowed_relationships = relationships,
            # node_properties = node_properties,
            prompt = self.prompt
        )

        def _convert_one(doc):
            return xform.convert_to_graph_documents([doc])

        graph_docs_all = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_convert_one, d) for d in documents]
            for f in as_completed(futures):
                graph_docs_all.extend(f.result())

        # 3) Write to Neo4j in batches (I/O-bound)
        for i in range(0, len(graph_docs_all), batch_size):
            batch = graph_docs_all[i:i+batch_size]
            self.graph.add_graph_documents(
                batch,
                baseEntityLabel = True,
                include_source = True,
            )
            if sleep_between_batches:
                time.sleep(sleep_between_batches)

    def create_fulltext_index(self):
        """
        Creates full text index entity
        Call after creating a graph
        """
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )

    def clear_graph(self):
        """Delete existing nodes and relationships"""
        self.graph.query("MATCH (n) DETACH DELETE n")

    def create_vector_index(self, embeddings):
        """Initialize vector index"""
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            search_type = "hybrid",
            node_label = "Document",
            text_node_properties = ["text"],
            embedding_node_property = "embedding",
            url = self.uri,
            username = self.username,
            password = self.password
        )
        return vector_index
    
    def create_entity_chain(self, llm):
        # Extract entities from text
        class Entities(BaseModel):
            """Identifying information about entities."""
            names: List[str] = Field(
                ...,
                description = "All the concept entities that appear in the text"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting concept entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following"
                    "input: {question}"
                )
            ]
        )

        entity_chain = prompt | llm.with_structured_output(Entities)
        return entity_chain

    @staticmethod
    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text
        search. It processes the input string by splitting it into words and 
        appending a similarity threshold (~2 changed characters) to each
        word, then combines them using the AND operator. Useful for mapping
        entities from user questions to database values, and allows for some 
        misspellings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    def structured_retriever(self, question: str, entity_chain):
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = entity_chain.invoke({"question": question})

        for entity in entities.names:
            response = self.graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN DISTINCT output
                LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(self, question: str, vector_index, entity_chain):
        # print(f"Search query: {question}")
        structured_data = self.structured_retriever(question, entity_chain)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data
    
    def query_chain(self, llm, vector_index, entity_chain):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        _context = RunnableLambda(
            lambda q: self.retriever(question = q, 
                                     vector_index = vector_index, 
                                     entity_chain = entity_chain)
        )

        chain = (
            RunnableParallel(
                {
                    "context": _context,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask_query(self, llm, chain, question: str):
        revised_question = self.rewrite_query(llm, question)
        self.chat_history.append({"user": revised_question})
        response = chain.invoke(revised_question)
        self.chat_history.append({"assistant": response})
        return response
    
    def rewrite_query(self, llm, question: str):
        """Rewrites question based on conversation history"""
        prompt = f"""
            You are a chatbot tutor for an educational learning platform.
            Rewrite the given question to incorporate any additional context from the user and chatbot's conversational history.
            The user's question might be a follow-up question, so your goal is to turn it into a more clear, concise, and comprehensive question.
            Do not infer anything that is not explicitly mentioned.
            Return the revised question, excluding new lines and excessive punctuation.
            User question: {question}
            Conversation history: {str(self.chat_history)}
        """
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))