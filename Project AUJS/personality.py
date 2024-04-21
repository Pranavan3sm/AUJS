from enum import Enum

class Personality(Enum):
    INFP = (
            "INFP : The Healer",
            "Focused on making the world a better place, INFPs are both idealists and perfectionists. INFPs possess strong value systems, are future-oriented, creative, and highly religious or philosophical individuals. Driven to meet the needs of others, INFPs tend to choose creative or human service-oriented careers that allow them to use their instinctive sense of empathy and remarkable communication skills. ",
            ["Creative Arts","Counseling/Psychotherapy","Social Work","Teaching","Healthcare Professions", "Coach", "College Professor","Composer", "Curator"]
        )
    ENFJ = (
            "ENFJ : The teacher",
            "ENFJs are people-oriented individuals who like generating opportunities to assist others in their long-term growth. They understand the perspectives of others and have the communication skills required in many professions, particularly teaching and counseling. ENFJs appreciate surroundings that provide variety, problem solving, a focus on the future, and opportunity to serve others and fulfill their humanitarian beliefs. ",
            ["Counseling/Psychology","Human Resources","Social Work","Teaching","Nonprofit Management", "Public Relations/Communications", "Healthcare Administration","Event Planning/Management", "Life Coaching","Conflict Resolution/Mediation"]
        )
    ENTJ = (
            "ENTJ  : The Commander",
            "ENTJs are natural-born leaders with strong desires to create order within organizations. They make decisions quickly, are very career-focused, and fit well into the corporate world. Since they like to be in charge—and need to be in charge to take advantage of their exceptional leadership skills—ENTJs are usually successful in the business world. Ideally, ENTJs should choose a career that allows them to creatively solve problems and enforce structure. ",
            ["Executive/Management Positions",
             "Entrepreneurship",
             "Business Consulting",
             "Management Consulting",
             "Engineering", 
             "Education Administration", 
             "Politics/Government"]
        )
    ENFP = (
            "ENFP : The Champion",
            "ENFPs are creative, energetic, outgoing, service-oriented individuals with well-developed verbal and written communication skills. ENFPs are the driving force that makes things happen and therefore are natural leaders. They work logically and rationally, and they are unique problem solvers. They are intuitive and perceptive about people and make good team players. Possessing a broad range of skills and talents, ENFPs often change careers several times. They seek out careers not confined by strict schedules and mundane tasks but provide a flexible environment that challenges them with new opportunities to see their innovations become reality. ",
            ["Creative Arts",
             "Public Relations/Marketing",
             "Entrepreneurship",
             "Counseling/Psychology",
             "Teaching/Training", 
             "Human Resources/Recruiting", 
             "Event Planning",
             "Social Work/Community Advocacy", 
             "Journalism/Media",
             "Consulting"]
        )
    ENTP = (
            "ENTP : The Visionary",
            "ENTPs are idea people with a wide range of capabilities and interests. Able to grasp difficult concepts and theories, ENTPs are logical thinkers and creative problem solvers. They are natural leaders capable of inspiring others to buy ideas. They often choose careers that allow them the personal freedom to use their creativity to generate new ideas and solve problems in a casual environment, where rules and restrictions are kept to a minimum.",
            ["Software Development",
             "Creative Arts",
             "Research",
             "Journalism",
             "Marketing", 
             "Entrepreneurship"]
        )
    ESFJ = (
            "ESFJ : The Provider",
            "ESFJs are warm, energetic, people persons. They are hard-working, organized, conscientious, and enjoy being part of a cooperative team. ESFJs are driven by a desire to lead and help people in practical ways. They often choose careers where they can observe others’ needs and organize a plan to meet those needs.",
            ["Healthcare Professional",
             "Teaching",
             "Human Resources Manager",
             "Event Planner",
             "Social Worker", 
             "Hospitality Management", 
             "Administrative Assistant",
             "Counselor or Therapist", 
             "Community Organizer",
             "Sales and Marketing"]
        )
    ESFP = (
            "ESFP : The Performer",
            "ESFPs thrive on new experiences and having a lot of contact with people. Possessing strong people skills, they tend to choose careers that allow them to help people in practical ways and often find themselves in the role of peacemaker. ESFPs enjoy work that is independent, resourceful, and allows them to have plenty of hands-on involvement and see the results of their efforts. They are great communicators, collaborators, and team players. ",
            ["Performing Arts",
             "Event Planning/Management",
             "Hospitality Industry",
             "Sales and Marketing",
             "Entertainment Industry", 
             "Teaching/Training", 
             "Fitness and Wellness",
             "Fashion and Design", 
             "Travel Industry"]
        )
    ESTJ = (
            "ESTJ : The Supervisor",
            "ESTJs focus on facts and concrete needs. They are analytical, conscientious, decisive, direct, efficient, responsible, and fact-minded individuals. ESTJs thrive in an environment built on order and continuity with explicit rules, expectations, and standards to follow. Leadership roles come easily for the ESTJ personality type, and they enjoy results-oriented work of a practical nature. ",
            ["Management or Leadership Roles",
             "Business Administration",
             "Law Enforcement or Military",
             "Financial Management",
             "Engineering", 
             "Education Administration", 
             "Healthcare Management",
             "Sales Management", 
             "Legal Professions",
             "Government Administration"]
        )
    ESTP = (
            "ESTP : The Promoter",
            "ESTPs are outgoing, enthusiastic doers, who live in a world of action. They have excellent people skills and the ability to quickly size up a situation and act accordingly. They have a special ability to react quickly in an emergency or crisis situation. ESTPs tend to choose careers that allow a lot of interaction with people in a fast-paced environment and do not require a lot of routine, detailed tasks. ESTPs are excellent sales people. ",
            ["Entrepreneurship",
             "Sales Representative",
             "Marketing Specialist",
             "Emergency Services",
             "Sports Management", 
             "Construction or Skilled Trades", 
             "Entertainment Industry",
             "Real Estate Agent", 
             "Military or Law Enforcement"]
        )
    INFJ = (
            "INFJ : The Counselor",
            "INFJs have great insight into people and situations. They are creative with deep feelings and strong convictions that guide their lives. Strongly humanitarian in outlook, INFJs tend to be idealists, and they are generally doers as well as dreamers. They often choose careers that allow them to use their inner vision, their ability to establish and maintain harmonious relationships, their creativity, and their strong oral and written communication skills.",
            ["Counselor/Therapist",
             "Psychologist",
             "Social Worker",
             "Writer/Author",
             "Artist/Designer", 
             "Teacher/Professor", 
             "History",
             "Human Resource Management", 
             "Interior Design"]
        )
    INTJ = (
            "INTJ : The Mastermind",
            "INTJs are strong individuals who seek new ways of looking at things. They are insightful and enjoy coming to new understandings. They often keep to themselves a great deal and are considered the most independent of all the 16 personality types. They are self-confident and prefer to look to the future rather than the past. They are builders of systems and appliers of theoretical models and focus on possibilities. They tend to prefer careers in which they are given a lot of autonomy. ",
            ["Strategic Planner/Management Consultant",
             "Scientist/Researcher",
             "Engineer/Software Developer",
             "Entrepreneur/Business Owner",
             "Financial Analyst/Investment Banker", 
             "Lawyer/Legal Consultant", 
             "Academic/Professor",
             "Project Manager"]
        )
    INTP = (
            "INTP : The Architect",
            "INTPs are architects of creative ideas and systems. Preferring theoretical rather than practical applications, INTPs love theory and abstract ideas. They are future-oriented and value truth, knowledge, and competence. INTPs usually prefer working in technical fields that provide the opportunities to apply logic to theories to find solutions and develop innovative approaches and systems. They prefer professionalism in a work environment that encourages freethinking, independence, and improvisation.",
            ["Scientist/Researcher",
             "Software Developer/Engineer",
             "Architect/Urban Planner",
             "Philosopher/Philosophy Professor",
             "Mathematician/Statistician", 
             "Writer/Content Creator", 
             "Consultant/Analyst",
             "University Professor/Researcher", 
             "Entrepreneur/Startup Founder",
             "Systems Analyst/IT Specialist"]
        )
    ISFJ = (
            "ISFJ : The Protector",
            "ISFJs, above all, desire to serve others. They are warm, kindhearted individuals, who bring an aura of quiet warmth, caring, and dependability to all that they do. They are hard-working and conscientious individuals prone to be quiet and serious. ISFJs tend to choose careers in which they can use their exceptional people-observation skills to determine what people want or need and then use their exceptional organizational abilities to create a structured plan to fulfill the need. Their excellent sense of space and function combined with their awareness of aesthetic quality also gives them special abilities in practical artistic endeavors, such as interior and fashion design.",
            ["Healthcare",
             "Education",
             "Social Work",
             "Counseling/Psychology",
             "Human Resources", 
             "Administrative Roles", 
             "Hospitality and Service Industry"]
        )
    ISFP = (
            "ISFP : The Composer",
            "ISFPs live in a world of sensational possibilities and are in tune with the way things look, taste, sound, feel, and smell. They have a strong aesthetic appreciation for art, and they are likely to be artists in some form because they are gifted at creating things that affect the senses. ISFPs are quiet, reserved, perceptive, and sympathetic individuals with little desire to lead or control others. They usually choose careers where they are given a great deal of autonomy so their natural artistic abilities can operate at their best and allow them to be consistent with their strong core of inner values. ISFPs are natural artists, counselors, and teachers. They rarely enjoy the fast-paced corporate world.",
            ["Graphic Designer/Visual Artist",
             "Fashion Designer/Stylist",
             "Interior Designer",
             "Photographer/Videographer",
             "Musician/Composer", 
             "Park Ranger/Outdoor Guide"]
        )
    ISTJ = (
            "ISTJ : The Inspector",
            "ISTJs are dependable and perseverant. They are quiet, serious, responsible, loyal individuals, who believe in time-honored laws and traditions. They possess a deep desire to promote security and peaceful living. Usually preferring to work alone, ISTJs do best in careers in which they can use their organizational skills and powers of concentration to create order and structure and therefore fit well into management positions.",
            ["Accountant/Financial Analyst",
             "Project Manager",
             "Engineer (Various Fields)",
             "Healthcare Administrator",
             "Quality Assurance/Compliance Officer", 
             "Logistics/Supply Chain Manager", 
             "Librarian/Archivist",
             "Human Resources Manager", 
             "Immigration and Customs Inspector",
             "Industrial Safety and Health Engineer"]
        )
    ISTP = (
            "ISTP : The Crafter",
            "With a compelling drive to understand how and why things work, ISTPs are action-oriented doers focused on the present. They are practical, realistic, and possess an excellent ability to apply logic and reason. They are masters of tools from the microscopic drill to the supersonic jet, and they are happiest in careers that allow them to use technical skills and tools to problem solve, troubleshoot, or manage a crisis. As leaders, ISTPs are usually up-front leading the charge.",
            ["Mechanic/Automotive Technician",
             "Engineer (Mechanical, Electrical, or Civil)",
             "Computer Programmer/Software Developer",
             "Pilot/Aircraft Mechanic",
             "Forensic Scientist", 
             "Emergency Medical Technician (EMT) or Paramedic", 
             "Construction Manager",
             "Freelancer/Consultant"]
        )




    def __init__(self, title, description, careers):
        self.title = title
        self.description = description
        self.careers = careers

    def __str__(self):
        return f"Personality {self.name}: ({self.title}, {self.description}, {self.careers})"
    
    @classmethod
    def from_string(cls, personality_str):
        try:
            return cls[personality_str.upper()]
        except KeyError:
            raise ValueError(f"No such color: {personality_str}")
