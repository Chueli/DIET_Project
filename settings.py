import sys
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_PATH = (f'model\{MODEL_NAME}')  if (sys.platform == "win32") else (f'model/{MODEL_NAME}')
DATA_PATH = 'data\\'
NUM_TOPICS = 10
NUM_WORDS = 4

diet_topics = [
    "Educational Technology History",
    "Multidisciplinary Approach",
    "User-Centered Design",
    "Design Thinking",
    "Design Methods",
    "Participatory Design",
    "Speculative Design",
    "Autobiographical Design",
    "UI/UX Principles",
    "Information Processing Model",
    "Cognitive Load Theory",
    "Behaviorism",
    "Cognitive Learning Theory",
    "Epistemology",
    "Dual-Coding Theory",
    "Spaced Repetition",
    "Multimedia Learning Theory",
    "Modality Principle",
    "Textual Modality",
    "Visual Modality",
    "Audio Modality",
    "NLP Approaches",
    "Symbolic NLP",
    "Statistical NLP",
    "Deep Learning NLP",
    "Large Language Models",
    "Multimodal Models",
    "Transformer Architecture",
    "Attention Mechanism",
    "Computer Vision",
    "Collaborative Learning",
    "Social Learning",
    "Pair Programming",
    "Peer Feedback",
    "Learnersourcing",
    "Collaboration Structure",
    "Coordination Awareness",
    "Client-Server Architecture",
    "Peer-to-Peer Architecture",
    "Federated Servers",
    "Conflict Resolution",
    "Operational Transform",
    "CRDT",
    "Intrinsic Motivation",
    "Extrinsic Motivation",
    "Expectancy-Value Theory",
    "Self-Determination Theory",
    "Social Cognitive Theory",
    "Behavioral Engagement",
    "Cognitive Engagement",
    "Emotional Engagement",
    "Gamification",
    "Game-Based Learning",
    "Personalized Learning",
    "Zone of Proximal Development",
    "Universal Design for Learning",
    "Adaptive Learning",
    "Learner Modeling",
    "Diagnostic Assessment",
    "Formative Assessment",
    "Summative Assessment",
    "Bloom's Taxonomy",
    "Concept Inventory",
    "Corrective Feedback",
    "Scaffolded Feedback"
]