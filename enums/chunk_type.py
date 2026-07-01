from enum import Enum

class ChunkType(Enum):
    NORMAL = "CharacterTextSplitter"
    RECURSIVE = "RecursiveCharacterTextSplitter"
    TOKEN = "TokenTextSplitter"