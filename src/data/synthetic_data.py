"""
Synthetic data generator for testing the RAG pipeline.
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.core.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generator for synthetic documents and queries."""

    def __init__(self):
        self.topics = [
            "artificial_intelligence",
            "machine_learning",
            "data_science",
            "cloud_computing",
            "cybersecurity",
            "blockchain",
            "web_development",
            "mobile_development",
            "devops",
            "database_management"
        ]

        self.document_templates = {
            "artificial_intelligence": [
                "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
                "Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
                "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
                "Computer vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos."
            ],
            "machine_learning": [
                "Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on new, unseen data.",
                "Unsupervised learning finds hidden patterns in data without labeled examples, including clustering and dimensionality reduction techniques.",
                "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving feedback.",
                "Feature engineering is the process of selecting, modifying, or creating variables that will be used in a machine learning model.",
                "Cross-validation is a technique used to assess how well a machine learning model will generalize to new, unseen data.",
                "Overfitting occurs when a machine learning model learns the training data too well and performs poorly on new data.",
                "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning models.",
                "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."
            ],
            "data_science": [
                "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data.",
                "Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics using statistical graphics.",
                "Data preprocessing is the process of cleaning and transforming raw data into a format suitable for analysis.",
                "Statistical inference is the process of using data analysis to deduce properties of an underlying probability distribution.",
                "Data visualization is the graphical representation of information and data using visual elements like charts, graphs, and maps.",
                "Big data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations.",
                "ETL (Extract, Transform, Load) is a data integration process that combines data from multiple sources into a single data store."
            ],
            "cloud_computing": [
                "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the internet.",
                "Infrastructure as a Service (IaaS) provides virtualized computing resources over the internet including servers, storage, and networking.",
                "Platform as a Service (PaaS) provides a platform allowing customers to develop, run, and manage applications without dealing with infrastructure.",
                "Software as a Service (SaaS) delivers software applications over the internet, on a subscription basis.",
                "Serverless computing is a cloud computing execution model where the cloud provider manages the infrastructure and automatically allocates resources.",
                "Containerization is a lightweight alternative to full machine virtualization that involves encapsulating an application with its dependencies.",
                "Microservices architecture is an approach to developing applications as a suite of small, independent services."
            ],
            "cybersecurity": [
                "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks and unauthorized access.",
                "Encryption is the process of converting information into a secret code to prevent unauthorized access.",
                "A firewall is a network security system that monitors and controls incoming and outgoing network traffic based on security rules.",
                "Penetration testing is a simulated cyber attack against a computer system to check for exploitable vulnerabilities.",
                "Multi-factor authentication (MFA) is a security system that requires more than one method of authentication to verify user identity.",
                "Malware is malicious software designed to damage, disrupt, or gain unauthorized access to computer systems.",
                "Zero-trust security is a security model that requires verification for every person and device trying to access a network."
            ],
            "blockchain": [
                "Blockchain is a distributed ledger technology that maintains a continuously growing list of records, called blocks, linked using cryptography.",
                "Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on blockchain technology.",
                "Smart contracts are self-executing contracts with terms directly written into code that run on blockchain networks.",
                "Decentralization refers to the distribution of functions, powers, and authority away from a central authority in blockchain networks.",
                "Consensus mechanisms are protocols that ensure all nodes in a blockchain network agree on the validity of transactions.",
                "Mining is the process of validating transactions and adding them to the blockchain in exchange for cryptocurrency rewards.",
                "Decentralized Autonomous Organizations (DAOs) are organizations run by smart contracts on blockchain networks."
            ],
            "web_development": [
                "Web development is the process of building and maintaining websites and web applications for the internet or intranet.",
                "Frontend development involves creating the user interface and user experience of websites using HTML, CSS, and JavaScript.",
                "Backend development involves server-side programming that handles database interactions, server logic, and API development.",
                "Responsive web design ensures websites work well on various devices and screen sizes using flexible layouts and media queries.",
                "RESTful APIs are architectural styles for designing networked applications based on stateless, client-server communication.",
                "Single Page Applications (SPAs) are web applications that load a single HTML page and dynamically update content.",
                "Progressive Web Apps (PWAs) are web applications that provide native app-like experiences using modern web technologies."
            ],
            "mobile_development": [
                "Mobile development is the process of creating software applications that run on mobile devices like smartphones and tablets.",
                "Native mobile development involves building apps specifically for one platform using platform-specific languages and tools.",
                "Cross-platform development allows developers to create apps that run on multiple mobile platforms using shared codebases.",
                "Mobile User Interface (UI) design focuses on creating intuitive and touch-friendly interfaces for mobile devices.",
                "Mobile app testing involves verifying app functionality, performance, and usability across different devices and operating systems.",
                "App Store Optimization (ASO) is the process of improving mobile app visibility and ranking in app store search results.",
                "Push notifications are messages sent by mobile apps to users' devices to engage and re-engage users."
            ],
            "devops": [
                "DevOps is a set of practices that combines software development and IT operations to shorten development lifecycle and provide continuous delivery.",
                "Continuous Integration (CI) is a practice where developers frequently integrate code changes into a shared repository.",
                "Continuous Deployment (CD) is a practice where code changes are automatically deployed to production after passing tests.",
                "Infrastructure as Code (IaC) is the practice of managing and provisioning computing infrastructure through machine-readable definition files.",
                "Containerization allows applications to run consistently across different environments by packaging them with their dependencies.",
                "Monitoring and logging are essential practices for tracking application performance and identifying issues in production.",
                "Version control systems track changes to code and enable collaboration among development teams."
            ],
            "database_management": [
                "Database management systems (DBMS) are software applications that interact with users, applications, and databases to capture and analyze data.",
                "Relational databases organize data into tables with rows and columns, using SQL for querying and manipulation.",
                "NoSQL databases provide flexible schema designs and are optimized for specific data models like documents, graphs, or key-value pairs.",
                "Database normalization is the process of organizing data to minimize redundancy and dependency in relational databases.",
                "Database indexing improves query performance by creating data structures that allow faster data retrieval.",
                "ACID properties (Atomicity, Consistency, Isolation, Durability) ensure reliable database transaction processing.",
                "Database backup and recovery strategies protect against data loss and ensure business continuity."
            ]
        }

        self.query_templates = {
            "artificial_intelligence": [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are the applications of AI?",
                "What is the difference between AI and machine learning?",
                "How is deep learning different from machine learning?",
                "What are neural networks?",
                "How does natural language processing work?",
                "What is computer vision?",
                "What are the ethical considerations in AI?",
                "What is the future of artificial intelligence?"
            ],
            "machine_learning": [
                "What is supervised learning?",
                "How does unsupervised learning work?",
                "What is reinforcement learning?",
                "What is feature engineering?",
                "How do you prevent overfitting?",
                "What is cross-validation?",
                "How does gradient descent work?",
                "What are the types of neural networks?",
                "What is model evaluation?",
                "How do you choose the right algorithm?"
            ],
            "data_science": [
                "What is data science?",
                "How do you perform exploratory data analysis?",
                "What is data preprocessing?",
                "What are the steps in a data science project?",
                "How do you handle missing data?",
                "What is statistical inference?",
                "How do you create effective data visualizations?",
                "What is big data?",
                "What is the ETL process?",
                "How do you measure model performance?"
            ],
            "cloud_computing": [
                "What is cloud computing?",
                "What are the types of cloud services?",
                "What is the difference between IaaS, PaaS, and SaaS?",
                "What is serverless computing?",
                "How does containerization work?",
                "What are microservices?",
                "What are the benefits of cloud computing?",
                "What are cloud security best practices?",
                "How do you choose a cloud provider?",
                "What is cloud migration?"
            ],
            "cybersecurity": [
                "What is cybersecurity?",
                "How does encryption work?",
                "What is a firewall?",
                "What is penetration testing?",
                "What is multi-factor authentication?",
                "What are the types of malware?",
                "What is zero-trust security?",
                "How do you secure a network?",
                "What are security best practices?",
                "How do you respond to a security incident?"
            ],
            "blockchain": [
                "What is blockchain?",
                "How does cryptocurrency work?",
                "What are smart contracts?",
                "What is decentralization?",
                "How do consensus mechanisms work?",
                "What is cryptocurrency mining?",
                "What are DAOs?",
                "What are the applications of blockchain?",
                "What are the challenges of blockchain?",
                "What is the future of blockchain?"
            ],
            "web_development": [
                "What is web development?",
                "What is frontend development?",
                "What is backend development?",
                "What is responsive web design?",
                "What are RESTful APIs?",
                "What are single page applications?",
                "What are progressive web apps?",
                "What are web development best practices?",
                "How do you optimize website performance?",
                "What are web security considerations?"
            ],
            "mobile_development": [
                "What is mobile development?",
                "What is native mobile development?",
                "What is cross-platform development?",
                "What is mobile UI design?",
                "How do you test mobile apps?",
                "What is app store optimization?",
                "How do push notifications work?",
                "What are mobile development best practices?",
                "How do you optimize mobile app performance?",
                "What are mobile security considerations?"
            ],
            "devops": [
                "What is DevOps?",
                "What is continuous integration?",
                "What is continuous deployment?",
                "What is infrastructure as code?",
                "How does containerization help DevOps?",
                "What is monitoring and logging?",
                "What is version control?",
                "What are DevOps best practices?",
                "How do you implement CI/CD?",
                "What are DevOps tools?"
            ],
            "database_management": [
                "What is database management?",
                "What are relational databases?",
                "What are NoSQL databases?",
                "What is database normalization?",
                "What is database indexing?",
                "What are ACID properties?",
                "What is database backup and recovery?",
                "How do you optimize database performance?",
                "What are database security best practices?",
                "What is database scaling?"
            ]
        }

    def generate_documents(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic documents."""
        documents = []

        for i in range(count):
            topic = random.choice(self.topics)
            templates = self.document_templates[topic]

            # Create a document with multiple paragraphs
            content_parts = random.sample(templates, min(random.randint(2, 4), len(templates)))
            content = "\n\n".join(content_parts)

            # Add some variation
            if random.random() < 0.3:
                content += f"\n\nAdditional considerations for {topic.replace('_', ' ')} include best practices, industry standards, and emerging trends in the field."

            document = {
                "id": str(uuid.uuid4()),
                "title": f"{topic.replace('_', ' ').title()} - Document {i+1}",
                "content": content,
                "topic": topic,
                "created_at": datetime.now() - timedelta(days=random.randint(1, 365)),
                "metadata": {
                    "source": "synthetic_generator",
                    "category": topic,
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                    "length": len(content),
                    "tags": [topic, random.choice(["tutorial", "overview", "guide", "reference"])],
                    "author": f"AI Assistant {random.randint(1, 10)}",
                    "version": f"1.{random.randint(0, 9)}"
                }
            }

            documents.append(document)

        logger.info(f"Generated {len(documents)} synthetic documents")
        return documents

    def generate_queries(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate synthetic queries."""
        queries = []

        for i in range(count):
            topic = random.choice(self.topics)
            templates = self.query_templates[topic]

            query_text = random.choice(templates)

            if random.random() < 0.2:
                query_text = f"Can you explain {query_text.lower()}"
            elif random.random() < 0.2:
                query_text = f"Tell me about {query_text.lower()}"

            query = {
                "id": str(uuid.uuid4()),
                "query": query_text,
                "topic": topic,
                "expected_sources": random.randint(1, 5),
                "created_at": datetime.now() - timedelta(hours=random.randint(1, 24)),
                "metadata": {
                    "source": "synthetic_generator",
                    "category": topic,
                    "complexity": random.choice(["simple", "moderate", "complex"]),
                    "intent": random.choice(["informational", "explanatory", "comparative", "procedural"])
                }
            }

            queries.append(query)

        logger.info(f"Generated {len(queries)} synthetic queries")
        return queries

    def generate_chat_history(self, query: str, topic: str) -> List[Dict[str, str]]:
        """Generate synthetic chat history for a query."""
        history = []

        if random.random() < 0.4:  # 40% chance of having chat history
            # Add 1-3 previous messages
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    # User message
                    user_queries = self.query_templates.get(topic, ["What can you tell me about this topic?"])
                    history.append({
                        "role": "user",
                        "content": random.choice(user_queries)
                    })
                else:
                    # Assistant response
                    responses = [
                        "I can help you with that. Let me explain...",
                        "That's a great question. Here's what you need to know...",
                        "Based on the information available, here's an overview...",
                        "Let me break that down for you..."
                    ]
                    history.append({
                        "role": "assistant",
                        "content": random.choice(responses)
                    })

        return history

    def save_to_files(self, documents: List[Dict[str, Any]], output_dir: str = "data/synthetic"):
        """Save synthetic documents to files."""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Save documents
        with open(f"{output_dir}/documents.json", "w") as f:
            json.dump(documents, f, indent=2, default=str)

        # Save individual text files
        for doc in documents:
            filename = f"{output_dir}/{doc['id']}.txt"
            with open(filename, "w") as f:
                f.write(f"Title: {doc['title']}\n")
                f.write(f"Topic: {doc['topic']}\n")
                f.write(f"Created: {doc['created_at']}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(doc['content'])

        logger.info(f"Saved {len(documents)} documents to {output_dir}")

    def create_sample_dataset(self, doc_count: int = 50, query_count: int = 20) -> Dict[str, Any]:
        """Create a complete sample dataset."""
        documents = self.generate_documents(doc_count)
        queries = self.generate_queries(query_count)

        for query in queries:
            query['chat_history'] = self.generate_chat_history(query['query'], query['topic'])

        dataset = {
            "documents": documents,
            "queries": queries,
            "metadata": {
                "generated_at": datetime.now(),
                "document_count": len(documents),
                "query_count": len(queries),
                "topics": self.topics,
                "generator_version": "1.0.0"
            }
        }

        return dataset


def generate_synthetic_documents(count: int = 50) -> List[Dict[str, Any]]:
    """Generate synthetic documents."""
    generator = SyntheticDataGenerator()
    return generator.generate_documents(count)


def generate_synthetic_queries(count: int = 20) -> List[Dict[str, Any]]:
    """Generate synthetic queries."""
    generator = SyntheticDataGenerator()
    return generator.generate_queries(count)


def create_sample_dataset(doc_count: int = 50, query_count: int = 20) -> Dict[str, Any]:
    """Create a complete sample dataset."""
    generator = SyntheticDataGenerator()
    return generator.create_sample_dataset(doc_count, query_count)