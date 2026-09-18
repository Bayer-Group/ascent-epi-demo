import json
import logging

from sqlalchemy import select

from ascent_domain.config import get_domain_settings
from ascent_domain.data_queries import data_queries
from ascent_domain.models.data_definitions import PersonalizedQuestion
from ascent_domain.omop.models.inference import prompts
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.llm.factory import create_assistant

logger = logging.getLogger(__name__)


async def generate_personalized_questions():
    personalized_questions_assistant = create_assistant(
        "gpt",
        model_name=get_domain_settings().OPENAI_EAST_US_GPT4O_MODEL_NAME,
        api_key=get_domain_settings().OPENAI_EAST_US_KEY,
        api_base=get_domain_settings().OPENAI_EAST_US_BASE,
        api_version=get_domain_settings().OPENAI_EAST_US_GPT4O_API_VERSION,
    )

    async with AsyncSessionLocal() as session:
        try:
            emails = await data_queries.get_unique_users_last_week(session)
            for user_email in emails:
                # Retrieve personalized questions for the current user from the database
                questions = await data_queries.get_user_questions(db=session, page=1, per_page=3, user=user_email)

                # Format search_text's as string
                last_questions = "\n".join(q.search_text for q in questions)

                roles = ""
                groups = ""

                # Replace the placeholders in the prompt with values
                personalized_questions_prompt_filled = prompts.personalized_questions_prompt.format(
                    roles=roles, therapeutic_area=groups, last_questions=last_questions
                )

                try:
                    response = await personalized_questions_assistant.get_response(personalized_questions_prompt_filled, json_format=True)
                except Exception as e:
                    # Handle exceptions from the assistant service, log the error, and continue
                    logger.error(f"Error getting response for user {user_email}: {e}")
                    continue

                questions_text = response if isinstance(response, str) else json.dumps(response)

                stmt = select(PersonalizedQuestion).where(PersonalizedQuestion.user_id == user_email)
                existing_question = (await session.scalars(stmt)).first()
                if existing_question:
                    existing_question.questions = questions_text
                    session.add(existing_question)
                else:
                    session.add(PersonalizedQuestion(user_id=user_email, questions=questions_text))

            # Commit the transaction after processing all users
            await session.commit()
        except Exception as e:
            # Handle any other exceptions that occur and log them
            logger.exception(f"An error occurred during question generation: {e}")
            # Optionally, you might want to roll back the transaction if an error occurs
            await session.rollback()
        finally:
            await session.close()  # Close database session
