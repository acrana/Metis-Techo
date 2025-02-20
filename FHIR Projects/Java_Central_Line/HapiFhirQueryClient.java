package com.example.fhir;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.rest.client.api.IGenericClient;
import org.hl7.fhir.r4.model.Bundle;
import org.hl7.fhir.r4.model.Device;

public class HapiFhirQueryClient {

    public static void main(String[] args) {
        // Create a FHIR context for R4
        FhirContext ctx = FhirContext.forR4();

        // Create a client that points to the HAPI FHIR Test/Demo Server
        IGenericClient client = ctx.newRestfulGenericClient("http://hapi.fhir.org/baseR4");

        // Perform a search for all Device resources on the server
        Bundle results = client.search()
                .forResource(Device.class)
                .returnBundle(Bundle.class)
                .execute();

        // Print the number of devices found and their IDs
        System.out.println("Found " + results.getEntry().size() + " Device resources.");
        for (Bundle.BundleEntryComponent entry : results.getEntry()) {
            Device device = (Device) entry.getResource();
            System.out.println("Device ID: " + device.getIdElement().getIdPart());
        }
    }
}
