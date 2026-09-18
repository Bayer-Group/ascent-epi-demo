# # Timeout for the medical coder api calls


class CodingType:
    STANDARD_CODING = "Standard"
    SOURCE_CODING = "Source"

    @classmethod
    def values(cls):
        return [cls.STANDARD_CODING, cls.SOURCE_CODING]
