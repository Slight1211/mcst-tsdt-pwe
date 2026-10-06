import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;

/** User-directed continuation of an existing M1L6 mesh with 15 eigenpairs. */
public class ContinueA75N15 {
  public static Model run() throws Exception {
    String input=System.getenv("MCST_JOINT_INPUT"),dir=System.getenv("MCST_JOINT_OUTPUT");
    if(input==null || dir==null || new File(dir+"/frequencies_remaining.csv").exists())
      throw new IllegalArgumentException("Invalid input/output");
    if(!new File(input).isFile())throw new FileNotFoundException("Missing MCST_JOINT_INPUT MPH checkpoint; historical checkpoints are omitted from the public release.");
    Model m=ModelUtil.load("JointA75N15",input);
    if(Math.abs(m.param().evaluate("a")-75e-6)>1e-14 || Math.abs(m.param().evaluate("hp")-24e-6)>1e-14)
      throw new IllegalArgumentException("Wrong geometry");
    if(Double.parseDouble(m.component("comp1").mesh("mesh1").feature("size1").getString("hauto"))!=1 ||
       Double.parseDouble(m.component("comp1").mesh("mesh1").feature("drySweep").feature("dis1").getString("numelem"))!=6 ||
       Double.parseDouble(m.sol("sol1").feature("e1").getString("neigs"))!=20)
      throw new IllegalArgumentException("Unexpected checkpoint settings");
    m.sol("sol1").feature("e1").set("neigs",15);
    System.out.println("CONTINUE_MESH mesh=1 layers=6 neigs=15 start_point=2");
    PrintWriter out=new PrintWriter(dir+"/frequencies_remaining.csv");
    try {
      out.println("point_index,kx_pi_over_a,ky_pi_over_a,mode_index,frequency_mhz,imag_frequency_mhz,vertical_fraction,core_fraction,core_vertical_fraction,solve_seconds");out.flush();
      for(int p=2;p<13;p++) {
        double x=p<=4?p/4.0:(p<=8?1.0:(12-p)/4.0),y=p<=4?0.0:(p<=8?(p-4)/4.0:(12-p)/4.0);
        String shift=p==12?"1[MHz]":"1.0E-4[MHz]";double tol=p==12?1e-9:1e-6;
        m.param().set("kx",x+"*pi/a");m.param().set("ky",y+"*pi/a");
        m.sol("sol1").feature("e1").set("shift",shift);m.sol("sol1").feature("e1").set("rtol",tol);
        m.sol("sol1").clearSolutionData();
        System.out.println("POINT_SETTINGS point="+p+" kx="+x+" ky="+y+" shift="+m.sol("sol1").feature("e1").getString("shift")+
          " rtol="+m.sol("sol1").feature("e1").getString("rtol")+" neigs="+m.sol("sol1").feature("e1").getString("neigs"));
        long start=System.nanoTime();m.sol("sol1").runAll();double seconds=(System.nanoTime()-start)/1e9;
        double[][] re=m.result().numerical("gevFreq").getReal(),im=m.result().numerical("gevFreq").getImag();
        if(re[0].length!=15)throw new IllegalArgumentException("Unexpected eigenpair count");
        for(int j=0;j<15;j++)out.println(p+","+x+","+y+","+(j+1)+","+re[0][j]+","+im[0][j]+","+re[1][j]+","+re[2][j]+","+re[3][j]+","+seconds);
        out.flush();if(out.checkError())throw new IOException("Raw export failed");
        m.save(dir+"/checkpoints/point_"+p+".mph");System.out.println("JOINT_POINT_COMPLETE point="+p+" seconds="+seconds);
      }
    } finally {out.close();}
    System.out.println("JOINT_REMAINING_COMPLETE");return m;
  }
  public static void main(String[] args)throws Exception{run();}
}
