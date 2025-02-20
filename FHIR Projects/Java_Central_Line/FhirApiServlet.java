package com.example.fhir;

import ca.uhn.fhir.context.FhirContext;
import ca.uhn.fhir.rest.server.IResourceProvider;
import ca.uhn.fhir.rest.server.RestfulServer;
import javax.servlet.annotation.WebServlet;
import java.util.ArrayList;
import java.util.List;

@WebServlet(urlPatterns = {"/fhir/*"})
public class FhirApiServlet extends RestfulServer {

    public FhirApiServlet() {
        super(FhirContext.forR4());
        List<IResourceProvider> providers = new ArrayList<>();
        providers.add(new CentralLineResourceProvider());
        setResourceProviders(providers);
    }
}
