package com.example.fhir;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.rest.client.api.IGenericClient;
import org.hl7.fhir.r4.model.Device;

public class HapiFhirReadDevice {
    public static void main(String[] args) {
        // Create a FHIR context for R4
        FhirContext ctx = FhirContext.forR4();
        
        // Create a client that points to the HAPI FHIR Test/Demo Server
        IGenericClient client = ctx.newRestfulGenericClient("http://hapi.fhir.org/baseR4");
        
        // Replace "234" with an actual Device resource ID you want to query.
        // For demonstration, we're using "234" because it was returned in our previous query.
        Device device = client.read()
                              .resource(Device.class)
                              .withId("234")
                              .execute();
        
        // Print the full Device resource as pretty-printed XML
        String deviceXml = ctx.newXmlParser().setPrettyPrint(true).encodeResourceToString(device);
        System.out.println(deviceXml);
    }
}
