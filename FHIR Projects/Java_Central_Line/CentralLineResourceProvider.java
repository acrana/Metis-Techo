package com.example.fhir;

import ca.uhn.fhir.rest.annotation.IdParam;
import ca.uhn.fhir.rest.annotation.Read;
import ca.uhn.fhir.rest.server.IResourceProvider;
import org.hl7.fhir.r4.model.DateTimeType;
import org.hl7.fhir.r4.model.Device;
import org.hl7.fhir.r4.model.Extension;
import org.hl7.fhir.r4.model.IdType;
import org.hl7.fhir.r4.model.StringType;

public class CentralLineResourceProvider implements IResourceProvider {

    @Override
    public Class<Device> getResourceType() {
        return Device.class;
    }

    // This method handles GET requests for a Device resource by ID.
    @Read
    public Device getDeviceById(@IdParam IdType theId) {
        Device device = new Device();
        // Set the device ID using the incoming parameter
        device.setId(theId.getIdPart());
        // Set a basic manufacturer
        device.setManufacturer("Central Line Manufacturer Inc.");

        // Add extension: Location where the central line was placed
        Extension locationExtension = new Extension();
        locationExtension.setUrl("http://example.org/fhir/StructureDefinition/central-line-location");
        locationExtension.setValue(new StringType("Right subclavian vein"));  // Example value
        device.addExtension(locationExtension);

        // Add extension: Date when the central line was inserted
        Extension dateInsertedExtension = new Extension();
        dateInsertedExtension.setUrl("http://example.org/fhir/StructureDefinition/central-line-date-inserted");
        dateInsertedExtension.setValue(new DateTimeType("2025-02-01T10:00:00Z"));  // Example date-time
        device.addExtension(dateInsertedExtension);

        // Add extension: Type of device used
        Extension deviceTypeExtension = new Extension();
        deviceTypeExtension.setUrl("http://example.org/fhir/StructureDefinition/central-line-device-type");
        deviceTypeExtension.setValue(new StringType("Double-lumen catheter"));  // Example type
        device.addExtension(deviceTypeExtension);

        // Add extension: Hygiene care instructions
        Extension hygieneExtension = new Extension();
        hygieneExtension.setUrl("http://example.org/fhir/StructureDefinition/central-line-hygiene-care");
        hygieneExtension.setValue(new StringType("Daily cleaning with chlorhexidine"));  // Example instruction
        device.addExtension(hygieneExtension);

        return device;
    }
}
