FHIR API Proof-of-Concept
This project is an early proof-of-concept demonstrating a basic FHIR API built using HAPI FHIR. It includes:

A custom RESTful server (using HAPI FHIR) to return dummy central line Device data.
A client that queries the HAPI FHIR public test server (http://hapi.fhir.org/baseR4) for Device resources.

Prerequisites
Java: JDK 11 or later (configured via JAVA_HOME)

Maven: For building the project

Apache Tomcat 9: For deploying the WAR file

A client class (HapiFhirQueryClient.java) is included to query the public HAPI FHIR Test/Demo Server.
