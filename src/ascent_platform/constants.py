

class ExpirationTTL:
    WEEK = 60 * 60 * 24 * 7
    DAY = 60 * 60 * 24
    MONTH = 60 * 60 * 24 * 30
    HOUR = 60 * 60

    @classmethod
    def values(cls):
        return [cls.WEEK, cls.DAY, cls.MONTH, cls.HOUR]


# Medical-coder HTTP bounds now live in ascent_platform.http.timeouts.


class CodingType:
    STANDARD_CODING = "Standard"
    SOURCE_CODING = "Source"

    @classmethod
    def values(cls):
        return [cls.STANDARD_CODING, cls.SOURCE_CODING]
